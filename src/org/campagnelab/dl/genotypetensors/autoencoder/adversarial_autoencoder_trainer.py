import torch
from torch.autograd import Variable

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar


class AdversarialAutoencoderTrainer(CommonTrainer):
    def init_model(self, create_model_function):
        super().init_model(create_model_function)
        # TODO: review list of params, names don't match between optimizers and source of parameters.
        self.encoder_semisup_generator_opt = torch.optim.SGD(self.net.encoder.parameters(), lr=self.args.lr,
                                                        momentum=self.args.momentum, weight_decay=self.args.L2)
        self.encoder_generator_opt = torch.optim.SGD(self.net.decoder.parameters(), lr=self.args.lr,
                                                momentum=self.args.momentum, weight_decay=self.args.L2)
        self.encoder_cat_opt = torch.optim.SGD(self.net.discriminator_cat.parameters(), lr=self.args.lr,
                                          momentum=self.args.momentum, weight_decay=self.args.L2)
        self.decoder_opt = torch.optim.SGD(self.net.discriminator_prior.parameters(), lr=self.args.lr,
                                      momentum=self.args.momentum, weight_decay=self.args.L2)
        self.optimizers=[self.encoder_semisup_generator_opt, self.encoder_generator_opt,self.encoder_cat_opt, self.decoder_opt]

    def train_semisup_aae(self, epoch,
                          performance_estimators=None):
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("train_loss")]
            performance_estimators += [FloatHelper("train_grad_norm")]
            print('\nTraining, epoch: %d' % epoch)
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()


        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unlabeled_loader = self.problem.unlabeled_loader()
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset, unlabeled_loader),
            is_cuda=self.use_cuda,
            batch_names=["training", "unlabeled"],
            requires_grad={"training": ["input"], "unlabeled": ["input"]},
            volatile={"training": ["metaData"], "unlabeled": []},
            recode_functions={
                "softmaxGenotype": recode_for_label_smoothing
            })

        for batch_idx, (_, data_dict) in enumerate(data_provider):
            input_s = data_dict["training"]["input"]
            input_u = data_dict["unlabeled"]["input"]
            target_s = data_dict["training"]["softmaxGenotype"]
            num_batches += 1
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            # net delegates to zub-modules:
            self.net.zero_grad()


    def test_semisup_aae(self, epoch, performance_estimators=None):
        print('\nTesting, epoch: %d' % epoch)
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("test_loss")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        for batch_idx, (_, data_dict) in enumerate(self.problem.validation_loader_range(0, self.args.num_validation)):
            inputs = data_dict["input"]
            if self.use_cuda:
                inputs = inputs.cuda()

            inputs, targets = Variable(inputs, volatile=True), Variable(inputs, volatile=True)

            # TODO: Obtain outputs properly from network
            outputs = None
            loss = self.criterion(outputs, targets)

            performance_estimators.set_metric_with_outputs(batch_idx, "test_loss", loss.data[0], outputs, targets)

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
