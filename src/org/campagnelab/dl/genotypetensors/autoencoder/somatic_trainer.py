from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, grad_norm


class SomaticTrainer(CommonTrainer):
    """Train an auto-encoder model from vec files."""


    def supervised_somatic(self, epoch,
                           performance_estimators=None):

        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("train_loss")]
            performance_estimators += [LossHelper("classification_loss")]
            performance_estimators += [LossHelper("frequency_loss")]
            performance_estimators += [FloatHelper("train_grad_norm")]
            print('\nTraining, epoch: %d' % epoch)

        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        cross_entropy_loss = CrossEntropyLoss()
        mse_loss = MSELoss()
        self.net.train()

        for batch_idx, (_, data_dict) in enumerate(train_loader_subset):
            inputs = data_dict["input"]
            is_mutated_base_target = data_dict["isBaseMutated"]
            # transform one-hot encoding into a class index:
            max,indices=is_mutated_base_target.max(dim=1)
            is_mutated_base_target=indices
            somatic_frequency_target = data_dict["somaticFrequency"]
            num_batches += 1

            if self.use_cuda:
                inputs, is_mutated_base_target, somatic_frequency_target = inputs.cuda(), \
                                                                           is_mutated_base_target.cuda(), \
                                                                           somatic_frequency_target.cuda()

            inputs, mut_targets, freq_targets = Variable(inputs), Variable(is_mutated_base_target, requires_grad=False), \
                                                Variable(somatic_frequency_target, requires_grad=False)
            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:

            self.optimizer_training.zero_grad()
            output_mut, output_frequency = self.net(inputs)

            classification_loss = cross_entropy_loss(output_mut, mut_targets)
            frequency_loss = mse_loss(output_frequency, freq_targets)
            optimized_loss = classification_loss + frequency_loss

            optimized_loss.backward()
            self.optimizer_training.step()
            performance_estimators.set_metric(batch_idx, "train_loss", optimized_loss.data[0])
            performance_estimators.set_metric(batch_idx, "classification_loss", classification_loss.data[0])
            performance_estimators.set_metric(batch_idx, "frequency_loss", frequency_loss.data[0])

            supervised_grad_norm = grad_norm(self.net.parameters())
            performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)

            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(["classification_loss", "frequency_loss"]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

        return performance_estimators


    def test_somatic_classifer(self, epoch, performance_estimators=None):
        print('\nTesting, epoch: %d' % epoch)
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("test_loss")]
            performance_estimators += [LossHelper("classification_loss")]
            performance_estimators += [LossHelper("frequency_loss")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        cross_entropy_loss = CrossEntropyLoss()
        mse_loss = MSELoss()
        for batch_idx, (_, data_dict) in enumerate(self.problem.validation_loader_range(0, self.args.num_validation)):
            inputs = data_dict["input"]
            is_mutated_base_target = data_dict["isBaseMutated"]
            # transform one-hot encoding into a class index:
            max, indices = is_mutated_base_target.max(dim=1)
            is_mutated_base_target = indices
            somatic_frequency_target = data_dict["somaticFrequency"]
            if self.use_cuda:
                inputs, is_mutated_base_target, somatic_frequency_target = inputs.cuda(), \
                                                                           is_mutated_base_target.cuda(), \
                                                                           somatic_frequency_target.cuda()

            inputs, mut_targets, freq_targets = Variable(inputs), Variable(is_mutated_base_target, volatile=True), \
                                                Variable(somatic_frequency_target, volatile=True)

            is_base_mutated, output_frequency = self.net(inputs)
            classification_loss = cross_entropy_loss(is_base_mutated, mut_targets)
            frequency_loss = mse_loss(output_frequency, freq_targets)
            test_loss = classification_loss + frequency_loss

            performance_estimators.set_metric(batch_idx, "test_loss", test_loss.data[0])
            performance_estimators.set_metric(batch_idx, "classification_loss", classification_loss.data[0])
            performance_estimators.set_metric(batch_idx, "frequency_loss", frequency_loss.data[0])

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

