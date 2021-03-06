import torch
from torch.nn import MultiLabelSoftMarginLoss

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


class GenotypingSupervisedTrainer(CommonTrainer):
    """Train a genotyping model using supervised training only."""
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
        self.class_reweighting=None

    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.criterion_classifier = MultiLabelSoftMarginLoss(weight=weights,reduce=False)
            self.class_reweighting=weights

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, previous_metric):
        return metric > previous_metric

    def train_supervised(self, epoch):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("supervised_loss")]
        performance_estimators += [AccuracyHelper("train_")]

        print('\nTraining, epoch: %d' % epoch)



        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset),
            device=self.device,
            batch_names=["training"],
            requires_grad={"training": ["input"]},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                "input": self.normalize_inputs
            },
            vectors_to_keep=["metaData"]
        )
        try:

            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["training"]["input"]
                target_s = data_dict["training"]["softmaxGenotype"]
                metadata = data_dict["training"]["metaData"]

                self.train_one_batch(performance_estimators,batch_idx, input_s,target_s,metadata)

                if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        return performance_estimators

    def train_one_batch(self, performance_estimators, batch_idx, input_s, target_s, metadata):
        # outputs used to calculate the loss of the supervised model
        # must be done with the model prior to regularization:
        self.net.train()
        batch_size=input_s.size(0)

        snp_weight = 1.0
        self.optimizer_training.zero_grad()
        self.net.zero_grad()
        output_s = self.net(input_s)
        output_s_p = self.get_p(output_s)
        _, target_index = torch.max(recode_as_multi_label(target_s), dim=1)
        supervised_loss = self.criterion_classifier(output_s, recode_as_multi_label(target_s))
        weighted_supervised_loss = self.reweight_loss(metadata, supervised_loss)

        optimized_loss = weighted_supervised_loss
        optimized_loss.backward()
        self.optimizer_training.step()
        performance_estimators.set_metric(batch_idx, "supervised_loss", weighted_supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", weighted_supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(
                             ["supervised_loss", "reconstruction_loss", "train_accuracy"]))

    def reweight_loss(self,  metadata, supervised_loss):
        indel_weight = self.args.indel_weight_factor
        variant_weight = self.args.variant_weight_factor
        # reweight loss for each class:
        supervised_loss = (supervised_loss * self.class_reweighting).sum(dim=1)
        # reweight for indels:
        is_indel = metadata[:, 1]
        is_indel_weights = is_indel.clone()
        is_indel_weights[is_indel == 0] = 1
        is_indel_weights[is_indel != 0] = indel_weight
        # reweight for variations/non-ref:
        is_variant = metadata[:, 0]
        is_variant_weights = is_variant.clone()
        is_variant_weights[is_variant == 0] = 1
        is_variant_weights[is_variant != 0] = variant_weight
        reweighting = is_indel_weights * is_variant_weights
        weighted_supervised_loss = (supervised_loss * reweighting).mean()
        return weighted_supervised_loss

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        output_s_exp = torch.exp(output_s)
        output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        return output_s_p

    def test_supervised(self, epoch):
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
            vectors_to_keep=["softmaxGenotype","metaData"]
        )
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["input"]
                target_s = data_dict["validation"]["softmaxGenotype"]
                metadata = data_dict["validation"]["metaData"]
                self.net.eval()
                self.test_one_batch(performance_estimators,batch_idx, input_s, target_s,errors=errors, metadata=metadata)

                if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                    break
            # print()
        finally:
            data_provider.close()
        print("test errors by class: ", str(errors))

        # Apply learning rate schedule:
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, (self.get_test_metric_name() +
                                         "must be found among estimated performance metrics")
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_metric, epoch)
        return performance_estimators

    def create_training_performance_estimators(self):
        performance_estimators = PerformanceList()
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

    def test_one_batch(self,performance_estimators, batch_idx, input_s, target_s,metadata=None, errors=None):
        if errors is None:
            errors = torch.zeros(target_s[0].size())

        output_s = self.net(input_s)
        output_s_p = self.get_p(output_s)

        supervised_loss = self.criterion_classifier(output_s, target_s)
        weighted_supervised_loss = self.reweight_loss( metadata, supervised_loss)

        self.estimate_errors(errors, output_s, target_s)
        _, target_index = torch.max(recode_as_multi_label(target_s), dim=1)
        _, output_index = torch.max(recode_as_multi_label(output_s_p), dim=1)
        performance_estimators.set_metric(batch_idx, "test_supervised_loss", weighted_supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", weighted_supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                  "test_accuracy"]))
