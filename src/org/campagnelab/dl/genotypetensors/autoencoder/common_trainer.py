import os
import sys

import torch
from torch.backends import cudnn
from torch.nn import MSELoss, CrossEntropyLoss, MultiLabelSoftMarginLoss

from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.LRHelper import LearningRateHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.LRSchedules import construct_scheduler
from org.campagnelab.dl.utils.utils import init_params, progress_bar


def _format_nice(n):
    try:
        if n == int(n):
            return str(n)
        if n == float(n):
            if n < 0.001:
                return "{0:.3E}".format(n)
            else:
                return "{0:.4f}".format(n)
    except ValueError:
        return str(n)


def recode_for_label_smoothing(one_hot_vector):
        epsilon = 0.2
        num_classes = len(one_hot_vector[0])
        one_hot_vector[one_hot_vector == 1] = 1.0 - epsilon
        one_hot_vector[one_hot_vector == 0] = epsilon / num_classes
        return one_hot_vector


def print_params(epoch, net):
    params = []
    for param in net.parameters():
        params += [p for p in param.view(-1).data]
    print("epoch=" + str(epoch) + " " + " ".join(map(str, params)))


class CommonTrainer:
    """
    Common code to train and test models and log their performance.
    """

    def __init__(self, args, problem, use_cuda):
        """
        Initialize model training with arguments and problem.

        :param args command line arguments.
        :param use_cuda When True, use the GPU.
         """

        self.model_label = "best"  # or latest
        self.max_regularization_examples = args.num_unlabeled if hasattr(args, "num_unlabeled") else 0
        self.max_validation_examples = args.num_validation if hasattr(args, "num_validation") else 0
        self.max_training_examples = args.num_training if hasattr(args, "num_training") else 0
        max_examples_per_epoch = args.max_examples_per_epoch if hasattr(args, 'max_examples_per_epoch') else None
        self.max_examples_per_epoch = (max_examples_per_epoch
                                       if max_examples_per_epoch is not None
                                       else self.max_regularization_examples)
        self.criterion = MSELoss()

        self.args = args
        self.problem = problem
        self.best_test_loss = sys.maxsize if not self.is_better(1, 0) else -1
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
        self.best_model = None
        self.agreement_loss = MSELoss()
        self.class_frequencies=None

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

        if hasattr(args, 'resume') and args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')

            assert os.path.isdir('models'), 'Error: no models directory found!'
            checkpoint = None
            try:
                checkpoint = torch.load('./models/pytorch_{}_{}.t7'.format(args.checkpoint_key, self.model_label))
            except FileNotFoundError:
                pass
            if checkpoint is not None:
                self.net = checkpoint['model']
                self.best_test_loss = checkpoint['best_test_loss']
                self.start_epoch = checkpoint['epoch']
                model_built = True
            else:
                print("Could not load model checkpoint, unable to --resume.")
                model_built = False

        if not model_built:
            print('==> Building model...')

            self.net = create_model_function(args.model, self.problem)
            self.net.apply(init_params)

        if self.use_cuda:
            self.net.cuda()
        cudnn.benchmark = True

        self.class_frequencies = self.class_frequency()
        print("class_frequency " + str(self.class_frequencies))

        self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum,
                                                  weight_decay=args.L2)

        self.scheduler_train = self.create_scheduler_for_optimizer(self.optimizer_training)

    def create_scheduler_for_optimizer(self,optimizer):
        return construct_scheduler(
            optimizer, "max" if self.is_better(1,0) else "min", factor=0.5,
            lr_patience=self.args.lr_patience if hasattr(self.args, 'lr_patience') else 10,
            ureg_reset_every_n_epoch=(self.args.reset_lr_every_n_epochs
                                      if hasattr(self.args, 'reset_lr_every_n_epochs')
                                      else None)
        )
    def log_performance_header(self, performance_estimators, kind="perfs"):
        if self.args.resume:
            return
        metrics = ["epoch", "checkpoint"]

        metrics += performance_estimators.metric_names()

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
        :param kind
        :return: a list of performance metrics corresponding to the epoch where test accuracy was maximum.
        """
        metrics = [epoch, self.args.checkpoint_key]
        metrics += performance_estimators.estimates_of_metrics()
        early_stop = False

        with open("all-{}-{}.tsv".format(kind, self.args.checkpoint_key), "a") as perf_file:
            perf_file.write(" ".join(map(_format_nice, metrics)))
            perf_file.write("\n")
        if self.best_performance_metrics is None:
            self.best_performance_metrics = performance_estimators

        metric = performance_estimators.get_metric(self.get_test_metric_name())
        if metric is not None and self.is_better(metric, self.best_test_loss):
            self.failed_to_improve = 0

            with open("best-{}-{}.tsv".format(kind, self.args.checkpoint_key), "a") as perf_file:
                perf_file.write(" ".join(map(_format_nice, metrics)))
                perf_file.write("\n")
            self.best_test_loss=metric

        self.save_model(metric, epoch, self.net, "latest")

        if metric is not None and (self.is_better(metric, self.best_test_loss) or metric == self.best_test_loss):
            self.save_checkpoint(epoch, metric)
            self.best_performance_metrics = performance_estimators
            self.best_model = self.net

        if metric is not None and self.is_better(metric, self.best_test_loss):
            self.failed_to_improve += 1
            if self.failed_to_improve > self.args.abort_when_failed_to_improve:
                print("We failed to improve for {} epochs. Stopping here as requested.")
                early_stop = True  # request early stopping

        return early_stop, self.best_performance_metrics

    def save_checkpoint(self, epoch, test_loss):
        # Save checkpoint.
        if test_loss < self.best_test_loss:
            print('Saving..')

            if self.best_model is not None:
                self.save_model(test_loss, epoch, self.best_model, "best")
                self.best_test_loss = test_loss
            else:
                # not best model, latest is best:
                self.save_model(test_loss, epoch, self.net, "best")


    def save_model(self, acc, epoch, model, model_label):
        model.eval()
        state = {
            'model': model.module if self.is_parallel else model,
            'best_test_loss': acc,
            'epoch': epoch,
        }
        additional_attrs = self.problem.model_attrs()
        if isinstance(additional_attrs, dict):
            for key in state:
                if key in additional_attrs:
                    del additional_attrs[key]
            state.update(additional_attrs)
        if not os.path.isdir('models'):
            os.mkdir('models')
        torch.save(state, './models/pytorch_{}_{}.t7'.format(self.args.checkpoint_key, model_label))

    def load_checkpoint(self, model_label="best"):
        if not os.path.isdir('models'):
            os.mkdir('models')
        state = torch.load('./models/pytorch_{}_{}.t7'.format(self.args.checkpoint_key, model_label))
        model = state['model']
        model.cpu()
        model.eval()
        return model

    def epoch_is_test_epoch(self, epoch):
        epoch_is_one_of_last_ten = epoch > (self.start_epoch + self.args.num_epochs - 10)
        return (epoch % self.args.test_every_n_epochs + 1) == 1 or epoch_is_one_of_last_ten

    def training_loops(self, training_loop_method=None, testing_loop_method=None):
        """Train the model in a completely supervised manner. Returns the performance obtained
           at the end of the configured training run.
        :return list of performance estimators that observed performance on the last epoch run.
        """
        assert training_loop_method is not None, "training_loop_method is required"
        assert testing_loop_method is not None, "testing_loop_method is required"
        header_written = False

        lr_train_helper = LearningRateHelper(scheduler=self.scheduler_train, learning_rate_name="train_lr")
        previous_test_perfs = None
        perfs = PerformanceList()

        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
            self.optimizer_training = torch.optim.SGD(self.net.parameters(), lr=self.args.lr,
                                                      momentum=self.args.momentum,
                                                      weight_decay=self.args.L2)
            perfs = PerformanceList()
            perfs += training_loop_method(epoch)

            perfs += [lr_train_helper]
            if previous_test_perfs is None or self.epoch_is_test_epoch(epoch):
                perfs += testing_loop_method(epoch)

            if not header_written:
                header_written = True
                self.log_performance_header(perfs)

            early_stop, perfs = self.log_performance_metrics(epoch, perfs)
            if early_stop:
                # early stopping requested.
                return perfs

        return perfs

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, best_test_loss):
        return metric > best_test_loss

    def rebuild_criterions(self, output_name, weights=None):
        """ This method set the criterions for the problem outputs. """
        pass

    def class_frequency(self, recode_as_multi_label=False):
        """
        Estimate class frequencies for the output vectors of the problem and rebuild criterions
        with weights that correct class imbalances.
        """

        train_loader_subset = self.problem.train_loader_subset_range(0, min(100000, self.args.num_training))
        data_provider = MultiThreadedCpuGpuDataProvider(iterator=zip(train_loader_subset), is_cuda=False,
                                                        batch_names=["training"],
                                                        volatile={"training": self.problem.get_vector_names()}
                                                        )

        class_frequencies = {}  # one frequency vector per output_name
        done = False
        for batch_idx, (_, data_dict) in enumerate(data_provider):
            if done:
                break
            for output_name in self.problem.get_output_names():
                target_s = data_dict["training"][output_name]
                if output_name not in class_frequencies.keys():
                    class_frequencies[output_name] = torch.ones(target_s[0].size())
                cf = class_frequencies[output_name]
                indices = torch.nonzero(target_s.data)
                for example_index in range(len(target_s)):
                    cf[indices[example_index, 1]] += 1

            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         "Class frequencies")

        for output_name in self.problem.get_output_names():

            class_frequencies_output = class_frequencies[output_name]
            # normalize with 1-f, where f is normalized frequency vector:
            weights = torch.ones(class_frequencies_output.size())
            weights -= class_frequencies_output / torch.norm(class_frequencies_output, p=1, dim=0)
            if self.use_cuda:
                weights = weights.cuda()

            self.rebuild_criterions(output_name=output_name, weights=weights)
        return class_frequencies

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        output_s_exp = torch.exp(output_s)
        output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        return output_s_p

    def estimate_batch_weight(self, meta_data, indel_weight, snp_weight):
        """Estimate a weight that gives more importance to batches with more indels in them.
        Helps optimization focus on indel predictions. Indel information is extracted
        from the meta_data field. """
        is_indel = meta_data.split(dim=1, split_size=1)[1]
        batch_size = len(is_indel)

        weights = is_indel.data.clone()
        weights[is_indel.data != 0] = indel_weight
        weights[is_indel.data == 0] = snp_weight
        batch_weight = torch.sum(weights) / batch_size
        return batch_weight