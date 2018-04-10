import os
import sys

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import MSELoss, CrossEntropyLoss, MultiLabelSoftMarginLoss

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer
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


class CommonHyperParamTrainer(CommonTrainer):
    """
    Common code to train and test models and log their performance.
    """

    def __init__(self, args_list, problem, use_cuda):
        """
        Initialize model training with arguments and problem.

        :param args List of command line arguments, one set of args per model to train.
        :param use_cuda When True, use the GPU.
         """
        # Use first argument for options common to all models:
        args=args_list[0]
        self.args_list=args_list
        super(CommonHyperParamTrainer, self).__init__(args,problem,use_cuda)
        self.model_list=[]
        self.optimizer_training_list=[]
        self.scheduler_train_list=[]
        self.best_test_loss_list=[]
        self.start_epoch_list=[]
        self.best_model_confusion_matrix_list=[]
        self.best_model_list=[]

    def init_model(self, create_model_function):
        """Resume training if necessary (args.--resume flag is True), or call the
        create_model_function to initialize a new model. This function must be called
        before train.

        The create_model_function takes one argument: the name of the model to be
        created.
        """

        for args in self.args_list:
            self.args=args
            super(CommonHyperParamTrainer, self).init_model(create_model_function)
            self.model_list+=[self.net]
            self.optimizer_training_list+=[self.optimizer_training]
            self.scheduler_train_list+=[self.scheduler_train]
            self.best_test_loss_list+=[self.best_test_loss]
            self.start_epoch_list+=[self.start_epoch]
            self.best_model_confusion_matrix_list+=[ self.best_model_confusion_matrix]
            self.best_model_list+=[self.best_model]

    def num_models(self):
        return len(self.model_list)

    def select_model(self, model_index):
        self.net += self.net[model_index]
        self.optimizer_training=self.optimizer_training_list[model_index]
        self.scheduler_train=self.scheduler_train_list[model_index]
        self.best_test_loss=self.best_test_loss_list [model_index]
        self.start_epoch=self.start_epoch_list[model_index]
        self.best_model_confusion_matrix=self.best_model_confusion_matrix_list[model_index]
        self.best_model=self.best_model_list[model_index]
