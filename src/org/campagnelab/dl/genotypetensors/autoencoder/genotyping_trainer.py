from torch.autograd import Variable

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, grad_norm


class GenotypingAutoEncoderTrainer(CommonTrainer):
    """Train an auto-encoder model from vec files."""

    def train_autoencoder(self, epoch,
                          performance_estimators=None):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("train_loss")]
            performance_estimators += [FloatHelper("train_grad_norm")]
            print('\nTraining, epoch: %d' % epoch)

        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)

        for batch_idx, (_, data_dict) in enumerate(train_loader_subset):
            inputs = data_dict["input"].to(self.device)
            num_batches += 1

            inputs, targets = Variable(inputs), Variable(inputs, requires_grad=False)
            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:
            self.net.train()
            self.optimizer_training.zero_grad()
            outputs = self.net(inputs)

            supervised_loss = self.criterion(outputs, targets)
            optimized_loss = supervised_loss
            optimized_loss.backward()
            self.optimizer_training.step()
            performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.item(),
                                                           outputs, targets)

            supervised_grad_norm = grad_norm(self.net.parameters())
            performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)

            performance_estimators.set_metric_with_outputs(batch_idx, "optimized_loss", optimized_loss.item(),
                                                           outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

        return performance_estimators

    def test_autoencoder(self, epoch, performance_estimators=None):
        print('\nTesting, epoch: %d' % epoch)
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("test_loss")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        for batch_idx, (_, data_dict) in enumerate(self.problem.validation_loader_range(0, self.args.num_validation)):
            inputs = data_dict["input"].to(self.device)

            inputs, targets = Variable(inputs, volatile=True), Variable(inputs, volatile=True)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            performance_estimators.set_metric_with_outputs(batch_idx, "test_loss", loss.item(), outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_loss"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()

        # Apply learning rate schedule:
        test_accuracy = performance_estimators.get_metric("test_loss")
        assert test_accuracy is not None, "test_loss must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_accuracy, epoch)
        return performance_estimators
