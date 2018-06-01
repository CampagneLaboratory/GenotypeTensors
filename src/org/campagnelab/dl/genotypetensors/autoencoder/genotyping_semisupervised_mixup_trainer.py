import torch
from torch.autograd import Variable
from torch.nn import MultiLabelSoftMarginLoss
from torchnet.meter import ConfusionMeter

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


class GenotypingSemisupervisedMixupTrainer(CommonTrainer):
    """Train a genotyping model using semisupervised mixup (labels on the unlabeled set are made up by sampling)."""

    def __init__(self, args, problem, device):
        super().__init__(args, problem, device)
        self.criterion_classifier = None
        self.cm=None
        if self.args.normalize:
            problem_mean = self.problem.load_tensor("input", "mean")
            problem_std = self.problem.load_tensor("input", "std")
        self.categorical_distribution=None
        self.normalize_inputs = lambda x: (normalize_mean_std(x, problem_mean=problem_mean,
                                                              problem_std=problem_std)
                                           if self.args.normalize
                                           else x)



    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.criterion_classifier = MultiLabelSoftMarginLoss(weight=weights)

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, previous_metric):
        return metric > previous_metric

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
        self.test_performance_estimators = performance_estimators
        return performance_estimators

    def train_semisupervised_mixup(self, epoch):
        performance_estimators = self.create_training_performance_estimators()

        print('\nTraining, epoch: %d' % epoch)



        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0

        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unlabeled_loader_subset = self.problem.unlabeled_loader()
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset, unlabeled_loader_subset),
            device=self.device,
            batch_names=["training", "unlabeled"],
            requires_grad={"training": ["input"], "unlabeled": ["input"]},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                "input": self.normalize_inputs
            },
            vectors_to_keep=["metaData"]
        )

        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s_1 = data_dict["training"]["input"]
                target_s_1 = data_dict["training"]["softmaxGenotype"]
                input_u_2 = data_dict["unlabeled"]["input"]
                metadata_1 = data_dict["training"]["metaData"]

                num_batches += 1

                self.train_one_batch(performance_estimators, batch_idx, input_s_1, target_s_1, metadata_1, input_u_2)

                if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        return performance_estimators

    def train_one_batch(self, performance_estimators, batch_idx, input_s_1, target_s_1, metadata_1, input_u_2):
        self.net.train()
        self.num_classes = len(target_s_1[0])
        genotype_frequencies = self.class_frequencies["softmaxGenotype"]

        category_prior = (genotype_frequencies / torch.sum(genotype_frequencies)).numpy()

        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0

        target_s_2 = self.dreamup_target_for(num_classes=self.num_classes,
                                             category_prior=category_prior,input=input_u_2).to(self.device)
        with self.lock:
            input_s_mixup, target_s_mixup = self._recreate_mixup_batch(input_s_1, input_u_2, target_s_1, target_s_2)

        self.optimizer_training.zero_grad()
        self.net.zero_grad()

        # outputs used to calculate the loss of the supervised model
        # must be done with the model prior to regularization:
        output_s = self.net(input_s_mixup)
        output_s_p = self.get_p(output_s)
        _, target_index = torch.max(target_s_mixup, dim=1)

        supervised_loss = self.criterion_classifier(output_s, target_s_mixup)
        # assume weight is the same for the two batches (we don't know metadata on the unlabeled batch):
        with self.lock:
            batch_weight = self.estimate_batch_weight(metadata_1, indel_weight=indel_weight,
                                                        snp_weight=snp_weight)

        supervised_loss = supervised_loss * batch_weight
        supervised_loss.backward()

        self.optimizer_training.step()
        performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(
                             ["supervised_loss", "reconstruction_loss", "train_accuracy"]))

    def reset_before_test_epoch(self):
        self.cm = ConfusionMeter(self.num_classes, normalized=False)

    def test_one_batch(self, performance_estimators,
                       batch_idx, input_s, target_s, metadata=None, errors=None):
        #if errors is None:
        #    errors = torch.zeros(target_s[0].size())

        output_s = self.net(input_s)
        output_s_p = self.get_p(output_s)

        _, target_index = torch.max(recode_as_multi_label(target_s), dim=1)
        _, output_index = torch.max(recode_as_multi_label(output_s_p), dim=1)
        self.cm.add(predicted=output_index.data, target=target_index.data)

        supervised_loss = self.criterion_classifier(output_s, target_s)
        #self.estimate_errors(errors, output_s_p, target_s)

        performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                  "test_accuracy"]))

    def test_semisupervised_mixup(self, epoch):
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
        if self.best_model is None:
            self.best_model=self.net
        self.reset_before_test_epoch()

        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["input"]
                target_s = data_dict["validation"]["softmaxGenotype"]
                self.net.eval()

                self.test_one_batch(performance_estimators, batch_idx, input_s, target_s, errors=None)


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

        self.compute_after_test_epoch()

        return performance_estimators

    def compute_after_test_epoch(self):
        """Call this method after an epoch of calling test_one_batch. This is used to compute
        variables in the trainer after each test epoch. """
        self.confusion_matrix = self.cm.value().transpose()

        if self.best_model_confusion_matrix is None:
            self.best_model_confusion_matrix = torch.from_numpy(self.confusion_matrix).to(self.device)