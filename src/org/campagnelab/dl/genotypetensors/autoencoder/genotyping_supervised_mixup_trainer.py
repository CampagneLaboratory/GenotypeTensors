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
        count_indices = list(to_binary(index.data[0], len(one_hot_vector)))
        for count_index in count_indices:
            coded[example_index, count_index] = 1
    # print(coded)
    return coded


class GenotypingSupervisedMixupTrainer(CommonTrainer):
    """Train a genotyping model using supervised training only."""
    def __init__(self, args, problem, use_cuda):
        super().__init__(args, problem, use_cuda)
        self.criterion_classifier = None
        if self.args.normalize:
            problem_mean = self.problem.load_tensor("input", "mean")
            problem_std = self.problem.load_tensor("input", "std")
        self.normalize_inputs = lambda x: (normalize_mean_std(x, problem_mean=problem_mean,
                                                              problem_std=problem_std)
                                           if self.args.normalize
                                           else x)

    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.criterion_classifier = MultiLabelSoftMarginLoss(weight=weights)

    def log_performance_metrics(self, epoch, performance_estimators, kind="perfs"):
        early_stop,best_performance_metrics=super().log_performance_metrics(epoch, performance_estimators, kind)

        # we load the best model we saved previously as a second model:
        self.best_model = self.load_checkpoint()
        if self.use_cuda:
            self.best_model = self.best_model.cuda()
        if self.confusion_matrix is not None:
            self.best_model_confusion_matrix = torch.from_numpy(self.confusion_matrix)
            if self.use_cuda:
                self.best_model_confusion_matrix = self.best_model_confusion_matrix.cuda()
        return (early_stop, best_performance_metrics)

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, previous_metric):
        return metric > previous_metric

    def _recreate_mixup_batch(self, input_1, input_2, target_1, target_2):
        assert input_1.size() == input_2.size(), ("Input 1 size {} does not equal input 2 size {} for mixup"
                                                  .format(input_1, input_2))
        assert target_1.size() == target_2.size(), ("Target 1 size {} does not equal target 2 size {} for mixup"
                                                    .format(target_1, target_2))
        assert input_1.size()[0] == target_1.size()[0], (
            "Input tensor has {} examples, target tensor has {} examples".format(input_1.size()[0],
                                                                                 input_2.size()[0])
        )
        lam = np.random.beta(self.args.mixup_alpha, self.args.mixup_alpha)
        input_mixup = lam * input_1 + (1.0 - lam) * input_2
        target_mixup = lam * target_1 + (1.0 - lam) * target_2
        return input_mixup, target_mixup

    def train_supervised_mixup(self, epoch):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("supervised_loss")]
        performance_estimators += [AccuracyHelper("train_")]

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
            is_cuda=self.use_cuda,
            batch_names=["training_1", "training_2"],
            requires_grad={"training_1": ["input"], "training_2": ["input"]},
            volatile={"training_1": ["metaData"], "training_2": ["metaData"]},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                "input": self.normalize_inputs
            }
        )

        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0
        for batch_idx, (_, data_dict) in enumerate(data_provider):
            input_s_1 = data_dict["training_1"]["input"]
            target_s_1 = data_dict["training_1"]["softmaxGenotype"]
            input_s_2 = data_dict["training_2"]["input"]
            target_s_2 = data_dict["training_2"]["softmaxGenotype"]
            metadata_1 = data_dict["training_1"]["metaData"]
            metadata_2 = data_dict["training_2"]["metaData"]
            num_batches += 1

            input_s_mixup, target_s_mixup = self._recreate_mixup_batch(input_s_1, input_s_2, target_s_1, target_s_2)

            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:

            self.optimizer_training.zero_grad()
            self.net.zero_grad()
            output_s = self.net(input_s_mixup)
            output_s_p = self.get_p(output_s)
            _, target_index = torch.max(target_s_mixup, dim=1)
            supervised_loss = self.criterion_classifier(output_s_p, target_s_mixup)

            batch_weight = self.estimate_batch_weight_mixup(metadata_1, metadata_2, indel_weight=indel_weight,
                                                            snp_weight=snp_weight)

            weighted_supervised_loss = supervised_loss * batch_weight
            optimized_loss = weighted_supervised_loss
            optimized_loss.backward()
            self.optimizer_training.step()
            performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.data[0])
            performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                                                           output_s_p, targets=target_index)

            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(
                             ["supervised_loss", "reconstruction_loss", "train_accuracy"]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break
        data_provider.close()

        return performance_estimators

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        output_s_exp = torch.exp(output_s)
        output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        return output_s_p

    def test_supervised_mixup(self, epoch):
        print('\nTesting, epoch: %d' % epoch)
        errors = None
        performance_estimators = PerformanceList()
        performance_estimators += [LossHelper("test_supervised_loss")]
        performance_estimators += [AccuracyHelper("test_")]

        self.net.eval()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset = self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(validation_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["validation"],
            requires_grad={"validation": []},
            volatile={
                "validation": ["input", "softmaxGenotype"]
            },
            recode_functions={
                "input": self.normalize_inputs
            }
        )

        for batch_idx, (_, data_dict) in enumerate(data_provider):
            input_s = data_dict["validation"]["input"]
            target_s = data_dict["validation"]["softmaxGenotype"]

            if errors is None:
                errors = torch.zeros(target_s[0].size())

            output_s = self.net(input_s)
            output_s_p = self.get_p(output_s)

            _, target_index = torch.max(recode_as_multi_label(target_s), dim=1)
            _, output_index = torch.max(recode_as_multi_label(output_s_p), dim=1)
            supervised_loss = self.criterion_classifier(output_s_p, target_s)
            self.estimate_errors(errors, output_s_p, target_s)

            performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.data[0])
            performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", supervised_loss.data[0],
                                                           output_s_p, targets=target_index)
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                  "test_accuracy"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()
        data_provider.close()
        print("test errors by class: ", str(errors))
        # Apply learning rate schedule:
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, (self.get_test_metric_name() +
                                         "must be found among estimated performance metrics")
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_metric, epoch)
        return performance_estimators
