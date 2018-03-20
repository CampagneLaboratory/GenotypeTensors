import numpy
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import MultiLabelSoftMarginLoss

import numpy as np
from torchnet.meter import ConfusionMeter
from random import random
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
        count_indices = list(to_binary(index.data[0], len(one_hot_vector)))
        for count_index in count_indices:
            coded[example_index, count_index] = 1
    # print(coded)
    return coded


class GenotypingSemisupervisedMixupTrainer(CommonTrainer):
    """Train a genotyping model using semisupervised mixup (labels on the unlabeled set are made up by sampling)."""

    def __init__(self, args, problem, use_cuda):
        super().__init__(args, problem, use_cuda)
        self.criterion_classifier = None
        if self.args.normalize:
            problem_mean = self.problem.load_tensor("input", "mean")
            problem_std = self.problem.load_tensor("input", "std")
        self.categorical_distribution=None
        self.normalize_inputs = lambda x: (normalize_mean_std(x, problem_mean=problem_mean,
                                                              problem_std=problem_std)
                                           if self.args.normalize
                                           else x)
        self.num_classes=0

    def get_category_sample(self, mini_batch_size, num_classes, category_prior, recode_labels=None):
        if self.categorical_distribution is None:
            # initialize a distribution to sample from num_classes with equal probability:
            if category_prior is None:
                category_prior = [1.0 / num_classes] * num_classes
            else:
                # Convert from numpy to a list of floats:
                category_prior = list(numpy.asarray(category_prior, dtype=float))
            self.categorical_distribution = Categorical(
                probs=numpy.reshape(torch.FloatTensor(category_prior * mini_batch_size),
                                    (mini_batch_size, num_classes)))
            self.categories_one_hot = torch.FloatTensor(mini_batch_size, num_classes)

        categories_as_int = self.categorical_distribution.sample().view(mini_batch_size, -1)
        self.categories_one_hot.zero_()
        self.categories_one_hot.scatter_(1, categories_as_int, 1)
        if recode_labels is not None:
            # apply label recode function:
            self.categories_one_hot = recode_labels(self.categories_one_hot)
        categories_real = Variable(self.categories_one_hot.clone(), requires_grad=False)
        return categories_real

    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.criterion_classifier = MultiLabelSoftMarginLoss(weight=weights)

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, previous_metric):
        return metric > previous_metric

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

    def train_semisupervised_mixup(self, epoch):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("supervised_loss")]
        performance_estimators += [AccuracyHelper("train_")]

        print('\nTraining, epoch: %d' % epoch)

        self.net.train()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0

        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unlabeled_loader_subset = self.problem.unlabeled_loader()
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset, unlabeled_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["training", "unlabeled"],
            requires_grad={"training": ["input"], "unlabeled": ["input"]},
            volatile={"training": ["metaData"], "unlabeled": ["metaData"]},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                "input": self.normalize_inputs
            }
        )
        genotype_frequencies = self.class_frequencies["softmaxGenotype"]
        category_prior = (genotype_frequencies / torch.sum(genotype_frequencies)).numpy()

        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0
        for batch_idx, (_, data_dict) in enumerate(data_provider):
            input_s_1 = data_dict["training"]["input"]
            target_s_1 = data_dict["training"]["softmaxGenotype"]
            input_s_2 = data_dict["unlabeled"]["input"]
            self.num_classes=len(target_s_1[0])
            target_s_2=self.dreamup_target_for(input=input_s_2,num_classes=self.num_classes,category_prior=category_prior)
            if self.use_cuda:
                target_s_2=target_s_2.cuda()
            metadata_1 = data_dict["training"]["metaData"]
            # assume metadata_2 is like metadata_1, since we don't know the indel status for the unlabeled set
            metadata_2=metadata_1
            num_batches += 1

            input_s_mixup, target_s_mixup = self._recreate_mixup_batch(input_s_1, input_s_2, target_s_1, target_s_2)

            # outputs used to calculate the loss of the supervised model
            # must be done with the model prior to regularization:

            self.optimizer_training.zero_grad()
            self.net.zero_grad()
            output_s = self.net(input_s_mixup)
            output_s_p = self.get_p(output_s)
            _, target_index = torch.max(target_s_mixup, dim=1)
            supervised_loss = self.criterion_classifier(output_s_p, target_s_mixup)

            batch_weight = self.estimate_batch_weight_mixup(metadata_1, metadata_2, indel_weight=indel_weight,
                                                            snp_weight=snp_weight)

            weighted_supervised_loss = supervised_loss * batch_weight
            optimized_loss = weighted_supervised_loss
            optimized_loss.backward()
            self.optimizer_training.step()
            performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.data[0])
            performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                                                           output_s_p, targets=target_index)

            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(
                             ["supervised_loss", "reconstruction_loss", "train_accuracy"]))

            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break
        data_provider.close()

        return performance_estimators

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        output_s_exp = torch.exp(output_s)
        output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        return output_s_p

    def test_semisupervised_mixup(self, epoch):
        print('\nTesting, epoch: %d' % epoch)
        errors = None
        performance_estimators = PerformanceList()
        performance_estimators += [LossHelper("test_supervised_loss")]
        performance_estimators += [AccuracyHelper("test_")]

        self.net.eval()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset = self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(validation_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["validation"],
            requires_grad={"validation": []},
            volatile={
                "validation": ["input", "softmaxGenotype"]
            },
            recode_functions={
                "input": self.normalize_inputs
            }
        )
        if self.best_model is None:
            self.best_model=self.net

        cm = ConfusionMeter(self.num_classes, normalized=False)
        for batch_idx, (_, data_dict) in enumerate(data_provider):
            input_s = data_dict["validation"]["input"]
            target_s = data_dict["validation"]["softmaxGenotype"]

            if errors is None:
                errors = torch.zeros(target_s[0].size())

            output_s = self.net(input_s)
            output_s_p = self.get_p(output_s)

            _, target_index = torch.max(recode_as_multi_label(target_s), dim=1)
            _, output_index = torch.max(recode_as_multi_label(output_s_p), dim=1)
            cm.add(predicted=output_index.data, target=target_index.data)

            supervised_loss = self.criterion_classifier(output_s_p, target_s)
            self.estimate_errors(errors, output_s_p, target_s)

            performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.data[0])
            performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", supervised_loss.data[0],
                                                           output_s_p, targets=target_index)
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                  "test_accuracy"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()
        data_provider.close()
        print("test errors by class: ", str(errors))
        # Apply learning rate schedule:
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, (self.get_test_metric_name() +
                                         "must be found among estimated performance metrics")
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_metric, epoch)
        self.confusion_matrix = cm.value().transpose()

        if self.best_model_confusion_matrix is None:
            self.best_model_confusion_matrix = torch.from_numpy(self.confusion_matrix)
            if self.use_cuda:
                self.best_model_confusion_matrix = self.best_model_confusion_matrix.cuda()
        return performance_estimators

    def dreamup_target_for(self, num_classes, category_prior,input):
        if self.best_model is None or self.args.label_strategy == "SAMPLING":
            return self.get_category_sample(self.mini_batch_size, num_classes=num_classes,
                                 category_prior=category_prior,
                                 recode_labels=lambda x: recode_for_label_smoothing(x, epsilon=self.epsilon))
        elif self.args.label_strategy == "VAL_CONFUSION":
            self.best_model.eval()
            # we use the best model we trained so far to predict the outputs. These labels will overfit to the
            # training set as training progresses:
            best_model_output = self.best_model(Variable(input.data, volatile=True))
            model_output_p=self.get_p(best_model_output).data
            # assume three classes and predicted= [0.9, 0.1, 0]
            # assume confusion matrix is [10, 10, 0,
            #                             80, 20, 10,
            #                              0, 30, 40]

            # reweight confusion matrix by probability of predictions:

            normalized_confusion_matrix=torch.renorm(self.best_model_confusion_matrix.type(torch.FloatTensor), p=1,
                                                     dim=0, maxnorm=1)
            if self.use_cuda:
                normalized_confusion_matrix = normalized_confusion_matrix.cuda()
                model_output_p=model_output_p.cuda()
            # result will be: [17,11,1,
            #                  40,25,25] corresponding to the marginals of the confusion matrix across the minibatch
            select =  (normalized_confusion_matrix.t()@model_output_p.t()).t()

            # Renormalize each row to get probability of each class according to the best model and validation confusion:
            targets2 = torch.renorm(select, p=1, dim=0, maxnorm=1)
            return Variable(targets2,requires_grad=False)
            # print("normalized: "+str(targets2))
        elif self.args.label_strategy == "VAL_CONFUSION_SAMPLING":
            # we use the best model we trained so far to predict the outputs. These labels will overfit to the
            # training set as training progresses:
            self.best_model.eval()
            best_model_output = self.best_model(Variable(input.data, volatile=True))
            model_output_p = self.get_p(best_model_output).data
            # assume three classes and predicted= [0.9, 0.1, 0]
            # assume confusion matrix is [10, 10, 0,
            #                             80, 20, 10,
            #                              0, 30, 40]

            # reweight confusion matrix by probability of predictions:
            normalized_confusion_matrix = torch.renorm(self.best_model_confusion_matrix.type(torch.FloatTensor), p=1,
                                                       dim=0, maxnorm=1)
            if self.use_cuda:
                normalized_confusion_matrix = normalized_confusion_matrix.cuda()
                model_output_p=model_output_p.cuda()

            # result will be: [17,11,1,
            #                  40,25,25] corresponding to the marginals of the confusion matrix across the minibatch
            select = (normalized_confusion_matrix.t() @ model_output_p.t()).t()

            # Renormalize each row to get probability of each class according to the best model and validation confusion:
            targets2 = torch.renorm(select, p=1, dim=0, maxnorm=1)
            # sample from this distribution:
            return self.get_category_sample(self.mini_batch_size, num_classes=num_classes,
                                 category_prior=targets2,
                                 recode_labels=lambda x: recode_for_label_smoothing(x, epsilon=self.epsilon))
