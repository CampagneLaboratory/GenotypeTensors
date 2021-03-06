import torch
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss

import numpy as np

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, normalize_mean_std


def to_binary(n, max_value):
    for index in list(range(max_value))[::-1]:
        yield 1 & int(n) >> index


enable_recode = False


def recode_as_multi_label(one_hot_vector):
    if not enable_recode:
        return one_hot_vector
    coded = torch.zeros(one_hot_vector.size())
    for example_index in range(0, len(one_hot_vector)):
        value, index = torch.max(one_hot_vector[example_index], dim=0)
        count_indices = list(to_binary(index.item(), len(one_hot_vector)))
        for count_index in count_indices:
            coded[example_index, count_index] = 1
    # print(coded)
    return coded


class GenotypingSupervisedMixupTrainer(CommonTrainer):
    """Train a genotyping model using supervised mixup training."""
    def __init__(self, args, problem, device):
        super().__init__(args, problem, device)
        self.criterion_classifier = None
        if self.args.normalize:
            problem_mean = self.problem.load_tensor("input", "mean")
            problem_std = self.problem.load_tensor("input", "std")
        self.normalize_inputs = lambda x: (normalize_mean_std(x, problem_mean=problem_mean,
                                                              problem_std=problem_std)
                                           if self.args.normalize
                                           else x)
    def create_training_performance_estimators(self):
        performance_estimators =PerformanceList()
        performance_estimators += [FloatHelper("supervised_loss")]
        performance_estimators += [AccuracyHelper("train_")]
        self.training_performance_estimators = performance_estimators
        return performance_estimators

    def create_test_performance_estimators(self):
        performance_estimators = PerformanceList()
        performance_estimators += [LossHelper("test_supervised_loss")]
        performance_estimators += [AccuracyHelper("test_")]
        self.test_performance_estimators=performance_estimators
        return performance_estimators

    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.criterion_classifier = MultiLabelSoftMarginLoss(weight=weights)

    def log_performance_metrics(self, epoch, performance_estimators, kind="perfs"):
        early_stop,best_performance_metrics=super().log_performance_metrics(epoch, performance_estimators, kind)

        # we load the best model we saved previously as a second model:
        self.best_model = self.load_checkpoint().to(self.device)
        if self.confusion_matrix is not None:
            self.best_model_confusion_matrix = torch.from_numpy(self.confusion_matrix).to(self.device)
        return (early_stop, best_performance_metrics)

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, previous_metric):
        return metric > previous_metric

    def train_supervised_mixup(self, epoch):
        performance_estimators = self.create_training_performance_estimators()

        print('\nTraining, epoch: %d' % epoch)

        self.net.train()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0

        train_loader_subset_1 = self.problem.train_loader_subset_range(0, self.args.num_training)
        train_loader_subset_2 = self.problem.train_loader_subset_range(0, self.args.num_training)
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset_1, train_loader_subset_2),
            device=self.device,
            batch_names=["training_1", "training_2"],
            requires_grad={"training_1": ["input"], "training_2": ["input"]},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                "input": self.normalize_inputs
            },
            vectors_to_keep=["metaData"]
        )


        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s_1 = data_dict["training_1"]["input"]
                target_s_1 = data_dict["training_1"]["softmaxGenotype"]
                input_s_2 = data_dict["training_2"]["input"]
                target_s_2 = data_dict["training_2"]["softmaxGenotype"]
                metadata_1 = data_dict["training_1"]["metaData"]
                metadata_2 = data_dict["training_2"]["metaData"]
                num_batches += 1

                self.train_one_batch(performance_estimators, batch_idx, input_s_1, input_s_2, target_s_1,target_s_2, metadata_1,metadata_2)

                if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        return performance_estimators

    def train_one_batch(self, performance_estimators, batch_idx, input_s_1, input_s_2,
                        target_s_1,target_s_2, metadata_1,metadata_2):
        input_s_mixup, target_s_mixup = self._recreate_mixup_batch(input_s_1, input_s_2, target_s_1, target_s_2)
        self.net.train()
        # outputs used to calculate the loss of the supervised model
        # must be done with the model prior to regularization:

        self.optimizer_training.zero_grad()
        self.net.zero_grad()
        output_s = self.net(input_s_mixup)
        output_s_p = self.get_p(output_s)
        _, target_index = torch.max(target_s_mixup, dim=1)
        supervised_loss = self.criterion_classifier(output_s, target_s_mixup)
        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0
        batch_weight = self.estimate_batch_weight_mixup(metadata_1, metadata_2, indel_weight=indel_weight,
                                                        snp_weight=snp_weight)

        weighted_supervised_loss = supervised_loss * batch_weight
        optimized_loss = weighted_supervised_loss
        optimized_loss.backward()
        self.optimizer_training.step()
        performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(
                             ["supervised_loss", "reconstruction_loss", "train_accuracy"]))

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        output_s_exp = torch.exp(output_s)
        output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        return output_s_p

    def test_supervised_mixup(self, epoch):
        print('\nTesting, epoch: %d' % epoch)
        errors = None
        performance_estimators = self.create_test_performance_estimators()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset = self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(validation_loader_subset),
            device=self.device,
            batch_names=["validation"],
            requires_grad={"validation": []},
            recode_functions={
                "input": self.normalize_inputs
            },
            vectors_to_keep=["softmaxGenotype"]
        )
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["input"]
                target_s = data_dict["validation"]["softmaxGenotype"]
                self.test_one_batch(performance_estimators,batch_idx, input_s, target_s, errors)

                if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                    break
            # print()
        finally:
            data_provider.close()
        print("test errors by class: ", str(errors))
        if self.reweight_by_validation_error:
            self.reweight_by_val_errors(errors)
        # Apply learning rate schedule:
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, (self.get_test_metric_name() +
                                         "must be found among estimated performance metrics")
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_metric, epoch)
        return performance_estimators


    def test_one_batch(self,performance_estimators, batch_idx, input_s, target_s,metadata=None, errors=None):
        if errors is None:
            errors = torch.zeros(target_s[0].size())
        self.net.eval()
        output_s = self.net(input_s)
        output_s_p = self.get_p(output_s)

        _, target_index = torch.max(recode_as_multi_label(target_s), dim=1)
        _, output_index = torch.max(recode_as_multi_label(output_s_p), dim=1)
        supervised_loss = self.criterion_classifier(output_s, target_s)
        #self.estimate_errors(errors, output_s_p, target_s)

        performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                  "test_accuracy"]))