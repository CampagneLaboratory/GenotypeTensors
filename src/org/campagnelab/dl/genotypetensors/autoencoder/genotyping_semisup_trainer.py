import torch
from torch.autograd import Variable
from torch.nn import MSELoss, MultiLabelSoftMarginLoss

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar


class GenotypingSemiSupTrainer(CommonTrainer):
    """Train a genotyping model using supervised and reconstruction on unlabeled set."""
    def __init__(self, args, problem, device):
        super().__init__(args, problem, device)
        self.criterion_classifier = None
        self.criterion_autoencoder = None

    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.criterion_classifier = MultiLabelSoftMarginLoss(weight=weights,size_average=False)
            self.criterion_autoencoder = MSELoss()

    def get_test_metric_name(self):
        return "test_supervised_loss"

    def is_better(self, metric, previous_metric):
        return metric < previous_metric

    def train_semisup(self, epoch):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("optimized_loss")]
        performance_estimators += [FloatHelper("supervised_loss")]
        performance_estimators += [FloatHelper("reconstruction_loss")]
        performance_estimators += [AccuracyHelper("train_")]

        print('\nTraining, epoch: %d' % epoch)

        self.net.train()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unlabeled_loader = self.problem.unlabeled_loader()
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset, unlabeled_loader),
            device=self.device,
            batch_names=["training", "unlabeled"],
            requires_grad={"training": ["input"], "unlabeled": ["input"]},
            recode_functions={"softmaxGenotype": lambda x: recode_for_label_smoothing(x,self.epsilon)},
            vectors_to_keep=["metaData"]
        )
        self.net.autoencoder.train()
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["training"]["input"]
                metadata = data_dict["training"]["metaData"]
                target_s = data_dict["training"]["softmaxGenotype"]
                input_u = data_dict["unlabeled"]["input"]
                num_batches += 1

                # need a copy of input_u and input_s as output:
                target_u = Variable(input_u.data, requires_grad=False)
                target_output_s = Variable(input_s.data, requires_grad=False)
                # outputs used to calculate the loss of the supervised model
                # must be done with the model prior to regularization:

                # Zero gradients:
                self.net.zero_grad()
                self.net.autoencoder.zero_grad()
                self.optimizer_training.zero_grad()

                output_s = self.net(input_s)
                output_u = self.net.autoencoder(input_u)
                input_output_s = self.net.autoencoder(input_s)
                output_s_p = self.get_p(output_s)

                _, target_index = torch.max(target_s, dim=1)
                supervised_loss = self.criterion_classifier(output_s, target_s)
                reconstruction_loss_unsup = self.criterion_autoencoder(output_u, target_u)
                reconstruction_loss_sup = self.criterion_autoencoder(input_output_s, target_output_s)
                reconstruction_loss = self.args.gamma * reconstruction_loss_unsup+reconstruction_loss_sup
                optimized_loss = supervised_loss + reconstruction_loss
                optimized_loss.backward()
                self.optimizer_training.step()
                performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.item())
                performance_estimators.set_metric(batch_idx, "reconstruction_loss", reconstruction_loss.item())
                performance_estimators.set_metric(batch_idx, "optimized_loss", optimized_loss.item())
                performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.item(),
                                                               output_s_p, targets=target_index)

                progress_bar(batch_idx * self.mini_batch_size,
                             self.max_training_examples,
                             performance_estimators.progress_message(["supervised_loss", "reconstruction_loss",
                                                                      "train_accuracy"]))

                if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        return performance_estimators

    def test_semi_sup(self, epoch):
        print('\nTesting, epoch: %d' % epoch)

        performance_estimators = PerformanceList()
        performance_estimators += [LossHelper("test_supervised_loss")]
        performance_estimators += [LossHelper("test_reconstruction_loss")]
        performance_estimators += [AccuracyHelper("test_")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset = self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(validation_loader_subset), device=self.device,
            batch_names=["validation"],
            requires_grad={"validation": []},
            vectors_to_keep=["input", "softmaxGenotype"]
        )
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["input"]
                target_s = data_dict["validation"]["softmaxGenotype"]
                # we need copies of the same tensors:
                input_u, target_u = Variable(input_s.data, volatile=True), Variable(input_s.data, volatile=True)

                output_s = self.net(input_s)
                output_u = self.net.autoencoder(input_u)
                output_s_p = self.get_p(output_s)

                _, target_index = torch.max(target_s, dim=1)

                supervised_loss = self.criterion_classifier(output_s, target_s)
                reconstruction_loss = self.criterion_autoencoder(output_u, target_u)

                performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.item())
                performance_estimators.set_metric(batch_idx, "test_reconstruction_loss", reconstruction_loss.item())
                performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", supervised_loss.item(),
                                                               output_s_p, targets=target_index)

                progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                             performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                      "test_accuracy"]))

                if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                    break
            # print()
        finally:
            data_provider.close()
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, self.get_test_metric_name() + "must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_metric, epoch)
        return performance_estimators
