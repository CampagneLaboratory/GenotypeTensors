import os

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MSELoss, MultiLabelSoftMarginLoss
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LRHelper import LearningRateHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.LRSchedules import construct_scheduler
from org.campagnelab.dl.utils.utils import grad_norm, progress_bar


def _format_nice(n):
    try:
        if n == int(n):
            return str(n)
        if n == float(n):
            if n < 0.001:
                return "{0:.3E}".format(n)
            else:
                return "{0:.4f}".format(n)
    except:
        return str(n)


def print_params(epoch, net):
    params = []
    for param in net.parameters():
        params += [p for p in param.view(-1).data]
    print("epoch=" + str(epoch) + " " + " ".join(map(str, params)))


class Trainer_AE:
    """Train an auto-encoder model from vec files."""

    def __init__(self, args, problem, use_cuda):
        """
        Initialize model training with arguments and problem.

        :param args command line arguments.
        :param use_cuda When True, use the GPU.
         """

        self.max_regularization_examples = args.num_unlabeled if hasattr(args, "num_unlabeled") else 0
        self.max_validation_examples = args.num_validation if hasattr(args, "num_validation") else 0
        self.max_training_examples = args.num_training if hasattr(args, "num_training") else 0
        max_examples_per_epoch = args.max_examples_per_epoch if hasattr(args, 'max_examples_per_epoch') else None
        self.max_examples_per_epoch = max_examples_per_epoch if max_examples_per_epoch is not None else self.max_regularization_examples
        self.criterion = MSELoss()

        self.args = args
        self.problem = problem
        self.best_acc = 0
        self.start_epoch = 0
        self.use_cuda = use_cuda
        self.mini_batch_size = problem.mini_batch_size()
        self.net = None
        self.optimizer_training = None
        self.scheduler_train = None
        self.unsuploader = self.problem.unlabeled_loader()
        self.trainloader = self.problem.train_loader()
        self.testloader = self.problem.validation_loader()
        self.is_parallel = False
        self.best_performance_metrics = None
        self.failed_to_improve = 0
        self.confusion_matrix = None
        self.best_model_confusion_matrix = None
        self.published_reconstruction_loss = False

    def init_model(self, create_model_function):
        """Resume training if necessary (args.--resume flag is True), or call the
        create_model_function to initialize a new model. This function must be called
        before train.

        The create_model_function takes one argument: the name of the model to be
        created.
        """
        args = self.args
        mini_batch_size = self.mini_batch_size
        # restrict limits to actual size of datasets:
        training_set_length = (len(self.problem.train_loader())) * mini_batch_size
        if hasattr(args, 'num_training') and args.num_training > training_set_length:
            args.num_training = training_set_length
        unsup_set_length = (len(self.problem.unlabeled_loader())) * mini_batch_size
        if hasattr(args, 'num_shaving') and args.num_shaving > unsup_set_length:
            args.num_shaving = unsup_set_length
        test_set_length = (len(self.problem.validation_loader())) * mini_batch_size
        if hasattr(args, 'num_validation') and args.num_validation > test_set_length:
            args.num_validation = test_set_length

        self.max_regularization_examples = args.num_shaving if hasattr(args, 'num_shaving') else 0
        self.max_validation_examples = args.num_validation if hasattr(args, 'num_validation') else 0
        self.max_training_examples = args.num_training if hasattr(args, 'num_training') else 0
        self.unsuploader = self.problem.unlabeled_loader()
        model_built = False
        self.best_performance_metrics = None
        self.best_model = None

        self.agreement_loss = MSELoss()

        if hasattr(args, 'resume') and args.resume:
            # Load checkpoint.

            print('==> Resuming from checkpoint..')

            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = None
            try:
                checkpoint = torch.load('./checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
            except                FileNotFoundError:
                pass
            if checkpoint is not None:

                self.net = checkpoint['net']
                self.best_acc = checkpoint['acc']
                self.start_epoch = checkpoint['epoch']
                self.best_model = checkpoint['best-model']
                self.best_model_confusion_matrix = checkpoint['confusion-matrix']
                model_built = True
            else:
                print("Could not load model checkpoint, unable to --resume.")
                model_built = False

        if not model_built:
            print('==> Building model {}'.format(args.model))

            self.net = create_model_function(args.model, self.problem)

        if self.use_cuda:
            self.net.cuda()
        cudnn.benchmark = True

        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                                  weight_decay=args.L2)

        self.scheduler_train = \
            construct_scheduler(self.optimizer_training, 'min', factor=0.5,
                                lr_patience=self.args.lr_patience if hasattr(self.args, 'lr_patience') else 10,
                                ureg_reset_every_n_epoch=self.args.reset_lr_every_n_epochs
                                if hasattr(self.args, 'reset_lr_every_n_epochs')
                                else None)

    def train(self, epoch,
              performance_estimators=None,
              train_supervised_model=True,
              ):

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

        for batch_idx, dict in enumerate(train_loader_subset):
            inputs=dict["input"]
            num_batches += 1

            if self.use_cuda:
                inputs = inputs.cuda()

            inputs, targets = Variable(inputs), Variable(inputs, requires_grad=False)
            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:
            self.net.train()
            self.optimizer_training.zero_grad()
            outputs = self.net(inputs)

            if train_supervised_model and self.args.mode == "supervised":

                supervised_loss = self.criterion(outputs, targets)
                optimized_loss = supervised_loss
                optimized_loss.backward()
                self.optimizer_training.step()
                performance_estimators.set_metric_with_outputs(batch_idx, "train_loss", supervised_loss.data[0],
                                                               outputs, targets)

            supervised_grad_norm = grad_norm(self.net.parameters())
            performance_estimators.set_metric(batch_idx, "train_grad_norm", supervised_grad_norm)

            performance_estimators.set_metric_with_outputs(batch_idx, "optimized_loss", optimized_loss.data[0],
                                                           outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

        return performance_estimators

    def test(self, epoch, performance_estimators=None):
        print('\nTesting, epoch: %d' % epoch)
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [LossHelper("test_loss")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        for batch_idx, dict in enumerate(self.problem.validation_loader_range(0, self.args.num_validation)):
            inputs=dict["input"]
            if self.use_cuda:
                inputs = inputs.cuda()

            inputs, targets = Variable(inputs, volatile=True), Variable(inputs, volatile=True)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            performance_estimators.set_metric_with_outputs(batch_idx, "test_loss", loss.data[0], outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_loss"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()

        # Apply learning rate schedule:
        test_accuracy = performance_estimators.get_metric("test_loss")
        assert test_accuracy is not None, "test_accuracy must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_accuracy, epoch)
        return performance_estimators

    def log_performance_header(self, performance_estimators, kind="perfs"):
        global best_test_loss
        if (self.args.resume):
            return

        metrics = ["epoch", "checkpoint"]

        metrics = metrics + performance_estimators.metric_names()

        if not self.args.resume:
            with open("all-{}-{}.tsv".format(kind, self.args.checkpoint_key), "w") as perf_file:
                perf_file.write(" ".join(map(str, metrics)))
                perf_file.write("\n")

            with open("best-{}-{}.tsv".format(kind, self.args.checkpoint_key), "w") as perf_file:
                perf_file.write(" ".join(map(str, metrics)))
                perf_file.write("\n")

    def log_performance_metrics(self, epoch, performance_estimators, kind="perfs"):
        """
        Log metrics and returns the best performance metrics seen so far.
        :param epoch:
        :param performance_estimators:
        :return: a list of performance metrics corresponding to the epoch where test accuracy was maximum.
        """
        metrics = [epoch, self.args.checkpoint_key]
        metrics = metrics + performance_estimators.estimates_of_metrics()

        early_stop = False
        with open("all-{}-{}.tsv".format(kind, self.args.checkpoint_key), "a") as perf_file:
            perf_file.write(" ".join(map(_format_nice, metrics)))
            perf_file.write("\n")
        if self.best_performance_metrics is None:
            self.best_performance_metrics = performance_estimators

        metric = performance_estimators.get_metric("test_accuracy")
        if metric is not None and metric > self.best_acc:
            self.failed_to_improve = 0

            with open("best-{}-{}.tsv".format(kind, self.args.checkpoint_key), "a") as perf_file:
                perf_file.write(" ".join(map(_format_nice, metrics)))
                perf_file.write("\n")

        if metric is not None and metric >= self.best_acc:

            self.save_checkpoint(epoch, metric)
            self.best_performance_metrics = performance_estimators
            if self.args.mode == "mixup":
                if self.args.two_models:
                    # we load the best model we saved previously as a second model:
                    self.best_model = self.load_checkpoint()
                    if self.use_cuda:
                        self.best_model = self.best_model.cuda(self.args.second_gpu_index)
                else:
                    self.best_model = self.net

                self.best_model_confusion_matrix = torch.from_numpy(self.confusion_matrix)
                if self.use_cuda:
                    self.best_model_confusion_matrix = self.best_model_confusion_matrix.cuda(self.args.second_gpu_index)

        if metric is not None and metric <= self.best_acc:
            self.failed_to_improve += 1
            if self.failed_to_improve > self.args.abort_when_failed_to_improve:
                print("We failed to improve for {} epochs. Stopping here as requested.")
                early_stop = True  # request early stopping

        return early_stop, self.best_performance_metrics

    def save_checkpoint(self, epoch, acc):

        # Save checkpoint.

        if acc > self.best_acc:
            print('Saving..')
            model = self.net
            model.eval()

            state = {
                'net': model.module if self.is_parallel else model,
                'best-model': self.best_model,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_{}.t7'.format(self.args.checkpoint_key))
            self.best_acc = acc

    def load_checkpoint(self):

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = torch.load('./checkpoint/ckpt_{}.t7'.format(self.args.checkpoint_key))
        model = state['net']
        model.cpu()
        model.eval()
        return model

    def epoch_is_test_epoch(self, epoch):
        epoch_is_one_of_last_ten = epoch > (self.start_epoch + self.args.num_epochs - 10)
        return (epoch % self.args.test_every_n_epochs + 1) == 1 or epoch_is_one_of_last_ten

    def training_supervised(self):
        """Train the model in a completely supervised manner. Returns the performance obtained
           at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()

        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):

            perfs = PerformanceList()

            perfs += self.train(epoch, train_supervised_model=True)

            perfs += [lr_train_helper]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += self.test(epoch)

            if (not header_written):
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs
