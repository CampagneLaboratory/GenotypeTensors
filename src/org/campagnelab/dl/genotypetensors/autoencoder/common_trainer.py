import os
import sys
import threading

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MSELoss, CrossEntropyLoss, MultiLabelSoftMarginLoss

from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.LRHelper import LearningRateHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.LRSchedules import construct_scheduler
from org.campagnelab.dl.utils.utils import init_params, progress_bar

import numpy as np
import numpy
import torch
from torch.autograd import Variable
from torch.distributions import Categorical


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


def recode_for_label_smoothing(one_hot_vector, epsilon=0.2):
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

    def __init__(self, args, problem, device):
        """
        Initialize model training with arguments and problem.

        :param args command line arguments.
        :param device where to put tensors and model- either cuda or cpu device
         """
        self.training_performance_estimators=self.create_training_performance_estimators()
        self.test_performance_estimators=self.create_test_performance_estimators()
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
        self.device = device
        self.mini_batch_size = problem.mini_batch_size()
        self.net = None
        self.optimizer_training = None
        self.scheduler_train = None
        self.lock=threading.Lock()
        self.is_parallel = False
        self.best_performance_metrics = None
        self.failed_to_improve = 0
        self.confusion_matrix = None
        self.best_model_confusion_matrix = None
        self.published_reconstruction_loss = False
        self.best_model = None
        self.agreement_loss = MSELoss()
        self.class_frequencies=None
        self.epsilon = args.epsilon_label_smoothing if hasattr( args,"epsilon_label_smoothing") else 0.0
        self.reweight_by_validation_error = args.reweight_by_validation_error if hasattr( args,"reweight_by_validation_error") else False
        self.num_classes = problem.output_size("softmaxGenotype")[0]
        self.estimate_errors_device = torch.device("cpu")

    def init_model(self, create_model_function,class_frequencies=None):
        """Resume training if necessary (args.--resume flag is True), or call the
        create_model_function to initialize a new model. This function must be called
        before train.

        The create_model_function takes one argument: the name of the model to be
        created.
        """
        args = self.args
        mini_batch_size = self.mini_batch_size
        # restrict limits to actual size of datasets:
        training_set_length = (len(self.problem.train_set()))
        if hasattr(args, 'num_training') and args.num_training > training_set_length:
            args.num_training = training_set_length
        unsup_set_length = (len(self.problem.unlabeled_set()))
        if hasattr(args, 'num_shaving') and args.num_shaving > unsup_set_length:
            args.num_shaving = unsup_set_length
        test_set_length = (len(self.problem.validation_set()))
        if hasattr(args, 'num_validation') and args.num_validation > test_set_length:
            args.num_validation = test_set_length

        self.max_regularization_examples = args.num_shaving if hasattr(args, 'num_shaving') else 0
        self.max_validation_examples = args.num_validation if hasattr(args, 'num_validation') else 0
        self.max_training_examples = args.num_training if hasattr(args, 'num_training') else 0

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
                self.best_model_confusion_matrix=checkpoint['confusion-matrix']
                # we just resumed from a best saved model, load it indei:
                self.best_model = self.load_checkpoint("best").to(self.device)
                model_built = True
            else:
                print("Could not load model checkpoint, unable to --resume.")
                model_built = False

        if not model_built:
            print('==> Building model...')

            self.net = create_model_function(args.model, self.problem).to(self.device)
            self.net.apply(init_params)

        cudnn.benchmark = True

        self.class_frequencies = self.class_frequency(class_frequencies=class_frequencies)
        #print("class_frequency " + str(self.class_frequencies))

        self.optimizer_training = self.get_default_optimizer_training(self.net,args.optimizer, args)

        self.scheduler_train = self.create_scheduler_for_optimizer(self.optimizer_training)

    def get_default_optimizer_training(self, model, optimizer_name, opt_args):
        if optimizer_name == "SGD":
            return torch.optim.SGD(model.parameters(), lr=opt_args.lr, momentum=opt_args.momentum,
                                   weight_decay=opt_args.L2)
        elif optimizer_name == "adagrad":
            return torch.optim.Adagrad(model.parameters(), lr=opt_args.lr, weight_decay=opt_args.L2)
        elif optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters(), lr=opt_args.lr, weight_decay=opt_args.L2)
        else:
            raise Exception("Unknown optimizer name: {}".format(optimizer_name))

    def set_common_lock(self, common_lock=None):
        self.lock=common_lock

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
        metrics += performance_estimators.estimates_of_metric()
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


        self.save_model(best_test_loss=metric, epoch=epoch, model=self.net, model_label="latest")
        if epoch==0:
            self.save_model(best_test_loss=metric, epoch=epoch, model=self.net, model_label="best")

        if metric is not None and not self.is_better(metric, self.best_test_loss):
            self.failed_to_improve += 1
            if self.failed_to_improve >= self.args.abort_when_failed_to_improve:
                print("We failed to improve for {} epochs. Stopping here as requested.".format(self.failed_to_improve))
                early_stop = True  # request early stopping

        if metric is not None and (self.is_better(metric, self.best_test_loss) or metric == self.best_test_loss):
            self.save_checkpoint(epoch, metric)
            self.best_performance_metrics = performance_estimators
            self.best_model = self.net


        if self.get_test_metric_name()=="test_accuracy" and metric is not None and \
            not self.is_better(metric, self.args.epoch_min_accuracy):
            early_stop=True
            print("Early stopping because accuracy is less than --epoch-min-accuracy {}".format(self.args.epoch_min_accuracy))
        return early_stop, self.best_performance_metrics

    def save_checkpoint(self, epoch, test_loss ):
        # Save checkpoint.
        if self.is_better(test_loss , self.best_test_loss):
            print('Saving..')

            if self.best_model is not None:
                self.save_model(test_loss, epoch, self.best_model, "best")
                self.best_test_loss = test_loss
                self.best_epoch =epoch
            else:
                # not best model, latest is best:
                self.save_model(test_loss, epoch, self.net, "latest")


    def save_model(self, best_test_loss, epoch, model, model_label):
        model.eval()
        state = {
            'model': model.module if self.is_parallel else model,
            'confusion-matrix': self.best_model_confusion_matrix,
            'best_test_loss': best_test_loss,
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
        return model

    def load_checkpoint_state(self, model_label="best"):
        if not os.path.isdir('models'):
            os.mkdir('models')
        state = torch.load('./models/pytorch_{}_{}.t7'.format(self.args.checkpoint_key, model_label))
        return state

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
        self.optimizer_training = self.get_default_optimizer_training(self.net, self.args.optimizer, self.args)

        for epoch in range(self.start_epoch, self.start_epoch + self.args.num_epochs):
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

    def class_frequency(self, recode_as_multi_label=False, class_frequencies=None):
        """
        Estimate class frequencies for the output vectors of the problem and rebuild criterions
        with weights that correct class imbalances.
        """
        if class_frequencies is None:

            train_loader_subset = self.problem.train_loader_subset_range(0, max(self.args.num_estimate_class_frequencies,
                                                                                min(100000, self.args.num_training)))
            data_provider = MultiThreadedCpuGpuDataProvider(
                iterator=zip(train_loader_subset),
                device=torch.device("cpu"),
                batch_names=["training"],
                vectors_to_keep=self.problem.get_vector_names()
            )

            class_frequencies = {}
            done = False
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                if batch_idx*self.mini_batch_size>self.args.num_estimate_class_frequencies:
                    break
                for output_name in self.problem.get_output_names():
                    target_s = data_dict["training"][output_name]
                    if output_name not in class_frequencies.keys():
                        class_frequencies[output_name] = torch.ones(target_s[0].size())
                    cf = class_frequencies[output_name]
                    indices = torch.nonzero(target_s.data)
                    indices = indices[:, 1]
                    for index in range(indices.size(0)):
                        cf[indices[index]] += 1

                progress_bar(batch_idx * self.mini_batch_size,
                             self.max_training_examples,
                             "Class frequencies")
        else:
            self.class_frequencies=class_frequencies
        for output_name in self.problem.get_output_names():

            class_frequencies_output = class_frequencies[output_name].to(self.device)
            # normalize with 1-f, where f is normalized frequency vector:
            weights = torch.ones(class_frequencies_output.size()).to(self.device)
            weights -= class_frequencies_output / torch.norm(class_frequencies_output, p=1, dim=0)

            self.rebuild_criterions(output_name=output_name, weights=weights)
        return class_frequencies

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        output_s_exp = torch.exp(output_s)
        output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        return output_s_p

    def sum_batch_weight(self, meta_data, indel_weight, snp_weight, index=1):
        is_indel = meta_data.split(dim=1, split_size=1)[index]
        weights = is_indel.data.clone()
        weights[is_indel.data != 0] = indel_weight
        weights[is_indel.data == 0] = snp_weight
        return len(is_indel), torch.sum(weights)

    def estimate_batch_weight(self, meta_data, indel_weight, snp_weight, index=1):
        """Estimate a weight that gives more importance to batches with more indels in them.
        Helps optimization focus on indel predictions. Indel information is extracted
        from the meta_data field. """
        batch_size, weight_sum = self.sum_batch_weight(meta_data, indel_weight, snp_weight,index)
        return weight_sum / batch_size

    def estimate_batch_weight_mixup(self, meta_data_1, meta_data_2, indel_weight, snp_weight):
        batch_size_1, weight_sum_1 = self.sum_batch_weight(meta_data_1, indel_weight, snp_weight)
        batch_size_2, weight_sum_2 = self.sum_batch_weight(meta_data_2, indel_weight, snp_weight)
        return (weight_sum_1 + weight_sum_2) / (batch_size_1 + batch_size_2)

    def reweight_by_val_errors(self, errors):
        weights = torch.ones(errors.size()).to(self.device)
        errors += 1
        weights -= errors / torch.norm(errors, p=1, dim=0)
        self.rebuild_criterions(output_name="softmaxGenotype", weights=weights)

    def estimate_errors(self, errors, output_s_p, target_s):
        _, target_index = torch.max(target_s, dim=1)
        _, output_index = torch.max(output_s_p, dim=1)
        target_index = target_index.to(self.estimate_errors_device)
        output_index = output_index.to(self.estimate_errors_device)
        for class_index in range(target_s[0].size()[0]):
            for example_index in range(self.mini_batch_size):
                if target_index[example_index].item() == class_index and \
                        (target_index[example_index].item() != output_index[example_index].item()):
                    errors[class_index] += 1

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

    def prepare_distribution(self, mini_batch_size, num_classes, category_prior, recode_labels=None):
        # initialize a distribution to sample from num_classes with equal probability:
        if category_prior is None:
            category_prior = [1.0 / num_classes] * num_classes
        else:
            # Convert from numpy to a list of floats:
            category_prior = list(numpy.asarray(category_prior, dtype=float))

        categorical_distribution = Categorical(
            probs=numpy.reshape(torch.FloatTensor(category_prior * mini_batch_size),
                                (mini_batch_size, num_classes)))
        return categorical_distribution

    def get_category_sample(self, mini_batch_size, num_classes, category_prior, recode_labels=None,
                            categorical_distribution=None):

        categorical_distribution = self.prepare_distribution(mini_batch_size, num_classes, category_prior,
                                                                 recode_labels)

        categories_one_hot = torch.zeros(mini_batch_size, num_classes)
        #categories_as_int=None
        with self.lock:
            categories_as_int = categorical_distribution.sample().view(mini_batch_size, -1)

        categories_one_hot.scatter_(1, categories_as_int, 1)
        if recode_labels is not None:
            # apply label recode function:
            categories_one_hot = recode_labels(categories_one_hot)

        categories_real = Variable(categories_one_hot, requires_grad=False)
        return categories_real

    def dreamup_target_for(self, num_classes, category_prior,input):
        strategy=self.args.label_strategy
        if self.best_model is None:
            strategy="SAMPLING"

        if  strategy == "UNIFORM" :
            category_prior=numpy.ones(self.num_classes)/self.num_classes
            return self.get_category_sample(self.mini_batch_size, num_classes=num_classes,
                                            category_prior=category_prior,
                                            recode_labels=lambda x: recode_for_label_smoothing(x, epsilon=self.epsilon))

        if strategy == "SAMPLING":

            return self.get_category_sample(self.mini_batch_size, num_classes=num_classes,
                                            category_prior=category_prior,
                                            recode_labels=lambda x: recode_for_label_smoothing(x, epsilon=self.epsilon))

        elif strategy ==  "VAL_CONFUSION":
            with self.lock:
                self.best_model.eval()
                # we use the best model we trained so far to predict the outputs. These labels will overfit to the
                # training set as training progresses:
                best_model_output = self.best_model(Variable(input.data, volatile=True).to(self.device))
                model_output_p=self.get_p(best_model_output).data.to(self.device)
                # assume three classes and predicted= [0.9, 0.1, 0]
                # assume confusion matrix is [10, 10, 0,
                #                             80, 20, 10,
                #                              0, 30, 40]

                # reweight confusion matrix by probability of predictions:

                normalized_confusion_matrix=torch.renorm(self.best_model_confusion_matrix.type(torch.FloatTensor), p=1,
                                                         dim=0, maxnorm=1).to(self.device)
                # result will be: [17,11,1,
                #                  40,25,25] corresponding to the marginals of the confusion matrix across the minibatch
                select =  (normalized_confusion_matrix.t()@model_output_p.t()).t()

                # Renormalize each row to get probability of each class according to the best model and validation confusion:
                targets2 = torch.renorm(select, p=1, dim=0, maxnorm=1)
                return Variable(targets2,requires_grad=False)
                # print("normalized: "+str(targets2))
        elif strategy == "VAL_CONFUSION_SAMPLING":
            with self.lock:
                # we use the best model we trained so far to predict the outputs. These labels will overfit to the
                # training set as training progresses:
                self.best_model.eval()
                best_model_output = self.best_model(Variable(input.data, volatile=True).to(self.device))
                model_output_p = self.get_p(best_model_output).data.to(self.device)
                # assume three classes and predicted= [0.9, 0.1, 0]
                # assume confusion matrix is [10, 10, 0,
                #                             80, 20, 10,
                #                              0, 30, 40]

                # reweight confusion matrix by probability of predictions:
                normalized_confusion_matrix = torch.renorm(self.best_model_confusion_matrix.type(torch.FloatTensor), p=1,
                                                           dim=0, maxnorm=1).to(self.device)

                # result will be: [17,11,1,
                #                  40,25,25] corresponding to the marginals of the confusion matrix across the minibatch
                select = (normalized_confusion_matrix.t() @ model_output_p.t()).t()

                # Renormalize each row to get probability of each class according to the best model and validation confusion:
                targets2 = torch.renorm(select, p=1, dim=0, maxnorm=1)
                result = torch.zeros(targets2.size())
                # sample from these distributions:
                for example in range(len(targets2)):
                    categorical_distribution = self.prepare_distribution(1, num_classes=num_classes,
                                                                         category_prior=targets2[example])
                    result[example] = self.get_category_sample(1, num_classes=num_classes,
                                                               categorical_distribution=categorical_distribution,
                                                               category_prior=None,
                                                               recode_labels=lambda x: recode_for_label_smoothing(x,
                                                                                                                  epsilon=self.epsilon)).data[
                        0]
                return Variable(result, requires_grad=False)

    def num_models(self):
        return 1

    def select_model(self, model_index):
        pass

    def create_test_performance_estimators(self):
        return None

    def create_training_performance_estimators(self):
        return None

    def reset_before_train_epoch(self):
        """Call this method before an epoch of calling train_one_batch. This is used to initialize
        variables in the trainer that need to reset before each training epoch. """
        pass

    def reset_before_test_epoch(self):
        """Call this method before an epoch of calling test_one_batch. This is used to initialize
        variables in the trainer that need to reset before each test epoch. """
        pass

    def compute_after_train_epoch(self):
        """Call this method after an epoch of calling train_one_batch. This is used to compute
        variables in the trainer after each train epoch. """
        pass

    def compute_after_test_epoch(self):
        """Call this method after an epoch of calling test_one_batch. This is used to compute
        variables in the trainer after each test epoch. """
        pass