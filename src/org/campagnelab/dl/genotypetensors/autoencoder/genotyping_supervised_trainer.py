import torch
from torch.autograd import Variable
from torch.legacy.nn import SoftMax
from torch.nn import MSELoss, BCELoss, BCEWithLogitsLoss, NLLLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss
from torch.nn.functional import softmax

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer
from org.campagnelab.dl.multithreading.sequential_implementation import DataProvider, CpuGpuDataProvider, \
    MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar


class GenotypingSupervisedTrainer(CommonTrainer):
    """Train a genotyping model using supervised training only."""
    def __init__(self, args, problem, use_cuda):
        super().__init__(args, problem, use_cuda)
        self.criterion_classifier = CrossEntropyLoss()

    def get_test_metric_name(self):
        return "test_supervised_loss"

    def is_better(self, metric, previous_metric):
        return metric< previous_metric

    def class_frequency(self):
        train_loader_subset = self.problem.train_loader_subset_range(0, min(100000,self.args.num_training))
        data_provider = MultiThreadedCpuGpuDataProvider(iterator=zip(train_loader_subset), is_cuda=False,
                                                        batch_names=["training"],
                                                        volatile={"training": ["input","softmaxGenotype"]})
        class_frequencies = None

        for batch_idx, dict in enumerate(data_provider):

            target_s = dict["training"]["softmaxGenotype"]
            if class_frequencies is None:
                class_frequencies=torch.zeros(target_s[0].size())
            max, target_index = torch.max(target_s, dim=1)

            class_frequencies[target_index.cpu().data] += 1
            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         "Class frequencies")


        class_frequencies=class_frequencies+1
        # normalize with 1-f, where f is normalized frequency vector:
        weights=torch.ones(class_frequencies.size())-(class_frequencies/torch.norm(class_frequencies, p=1, dim=0))
        if self.use_cuda:
            weights=weights.cuda()
        self.criterion_classifier = CrossEntropyLoss(weight=weights)
        return class_frequencies

    def train_supervised(self, epoch):

        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("supervised_loss")]

        print('\nTraining, epoch: %d' % epoch)

        self.net.train()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        data_provider = MultiThreadedCpuGpuDataProvider(iterator=zip(train_loader_subset),is_cuda=self.use_cuda,
                                     batch_names=["training"],
                                     requires_grad={"training": ["input"]},
                                     volatile={"training": [] })

        self.net.autoencoder.train()
        for batch_idx, dict in enumerate(data_provider):
            input_s = dict["training"]["input"]
            target_s = dict["training"]["softmaxGenotype"]


            num_batches += 1

            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:

            self.optimizer_training.zero_grad()
            self.net.zero_grad()
            output_s = self.net(input_s)
            output_s_p = self.get_p(output_s)
            max, target_index= torch.max(target_s, dim=1)
            supervised_loss = self.criterion_classifier(output_s_p,target_index )
            optimized_loss = supervised_loss
            optimized_loss.backward()
            self.optimizer_training.step()
            performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.data[0])

            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(["supervised_loss", "reconstruction_loss"]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break
        data_provider.close()

        return performance_estimators

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        #output_s_exp = torch.exp(output_s)
        #output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        output_s_p= softmax(output_s)
        return output_s_p

    def test_supervised(self, epoch):
        print('\nTesting, epoch: %d' % epoch)
        errors = None
        performance_estimators = PerformanceList()
        performance_estimators += [LossHelper("test_supervised_loss")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset=self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = MultiThreadedCpuGpuDataProvider(iterator=zip(validation_loader_subset), is_cuda=self.use_cuda,
                                     batch_names=["validation"],
                                     requires_grad={"validation": []},
                                     volatile={"validation": ["input","softmaxGenotype"]})
        for batch_idx, dict in enumerate(data_provider):
            input_s = dict["validation"]["input"]
            target_s = dict["validation"]["softmaxGenotype"]

            if errors is None:
                errors=torch.zeros(target_s[0].size())

            output_s = self.net(input_s)
            output_s_p = self.get_p(output_s)

            max, target_index = torch.max(target_s, dim=1)
            max, output_index = torch.max(output_s_p, dim=1)
            supervised_loss = self.criterion_classifier(output_s_p, target_index)
            errors[target_index.cpu().data] += torch.ne(target_index.cpu().data, output_index.cpu().data).type(torch.FloatTensor)

            performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.data[0])
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss","test_reconstruction_loss"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()
        data_provider.close()
        print("test errors by class: ", str(errors))
        # Apply learning rate schedule:
        test_accuracy = performance_estimators.get_metric("test_supervised_loss")
        assert test_accuracy is not None, "test_supervised_loss must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_accuracy, epoch)
        return performance_estimators
