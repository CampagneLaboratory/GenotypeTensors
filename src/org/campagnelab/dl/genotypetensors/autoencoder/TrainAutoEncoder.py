'''Train an auto-encoder for .vec/vecp files.'''
from __future__ import print_function

import argparse
import random
import string
import sys

import torch

# MIT License
#
# Copyright (c) 2017 Fabien Campagne
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from org.campagnelab.dl.genotypetensors.autoencoder.autoencoder import create_autoencoder_model
from org.campagnelab.dl.genotypetensors.autoencoder.genotyping_trainer import GenotypingAutoEncoderTrainer
from org.campagnelab.dl.genotypetensors.autoencoder.sbi_classifier import create_classifier_model
from org.campagnelab.dl.genotypetensors.autoencoder.somatic_trainer import SomaticTrainer
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an auto-encoder for .vec files.')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint.')

    parser.add_argument('--constant-learning-rates', action='store_true',
                        help='Use constant learning rates, not schedules.')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=128)
    parser.add_argument('--encoded-size', type=int,
                        help='Size the auto-encoder compresses the input to. The number of floats used '
                             'to represent the encoded data.', default=32)
    parser.add_argument('--num-epochs', '--max-epochs', type=int,
                        help='Number of epochs to run before stopping. Additional epochs when --resume.', default=200)
    parser.add_argument('--num-training', '-n', type=int, help='Maximum number of training examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--num-validation', '-x', type=int, help='Maximum number of training examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--num-unlabeled', '-u', type=int, help='Maximum number of unlabeled examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--max-examples-per-epoch', type=int, help='Maximum number of examples scanned in an epoch'
                                                                   '(e.g., for ureg model training). By default, equal to the '
                                                                   'number of examples in the training set.',
                        default=None)
    parser.add_argument('--momentum', type=float, help='Momentum for SGD.', default=0.9)
    parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)
    parser.add_argument('--seed', type=int,
                        help='Random seed', default=random.randint(0, sys.maxsize))
    parser.add_argument('--checkpoint-key', help='random key to save/load checkpoint',
                        default=''.join(random.choices(string.ascii_uppercase, k=5)))
    parser.add_argument('--lr-patience', default=10, type=int,
                        help='number of epochs to wait before applying LR schedule when loss does not improve.')
    parser.add_argument('--model', default="PreActResNet18", type=str,
                        help='The model to instantiate. One of VGG16,	ResNet18, ResNet50, ResNet101,ResNeXt29, ResNeXt29, DenseNet121, PreActResNet18, DPN92')
    parser.add_argument('--problem', default="genotyping:basename", type=str,
                        help='The genotyping problem dataset name. basename is used to locate file named basename-train.vec, basename-validation.vec, basename-unlabeled.vec')
    parser.add_argument('--mode', help='Training mode: autoencoder, supervised_somatic',
                        default="supervised")
    parser.add_argument("--reset-lr-every-n-epochs", type=int,
                        help='Reset learning rate to initial value every n epochs.')
    parser.add_argument('--abort-when-failed-to-improve', default=sys.maxsize, type=int,
                        help='Abort training if performance fails to improve for more than the specified number of epochs.')
    parser.add_argument('--test-every-n-epochs', type=int,
                        help='Estimate performance on the test set every n epochs. '
                             'Note that when test is skipped, the previous test '
                             'performances are reported in the log until new ones are available.'
                             'This parameter does not affect testing for the last 10 epochs of a run, each test is '
                             'performed for these epochs.', default=1)

    args = parser.parse_args()

    if args.max_examples_per_epoch is None:
        args.max_examples_per_epoch = args.num_training

    print("Executing " + args.checkpoint_key)

    with open("args-{}".format(args.checkpoint_key), "w") as args_file:
        args_file.write(" ".join(sys.argv))

    use_cuda = torch.cuda.is_available()
    is_parallel = False
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    problem=None
    if args.problem.startswith("genotyping:"):
        problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem)
    elif args.problem.startswith("somatic:"):
        problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)


    def get_metric_value(all_perfs, query_metric_name):
        for perf in all_perfs:
            metric = perf.get_metric(query_metric_name)
            if metric is not None:
                return metric


    def train_once(args, problem, use_cuda):
        problem.describe()
        training_loop_method = None
        testing_loop_method = None
        if args.mode == "autoencoder":

            model_trainer = GenotypingAutoEncoderTrainer(args=args, problem=problem, use_cuda=use_cuda)
            model_trainer.init_model(create_model_function=
                                     (lambda model_name, problem: create_autoencoder_model(model_name,
                                                                                           problem,
                                                                                           encoded_size=args.encoded_size)))
            training_loop_method = model_trainer.train_autoencoder
            testing_loop_method = model_trainer.test_autoencoder

        elif args.mode == "supervised_somatic":
            model_trainer = SomaticTrainer(args=args, problem=problem, use_cuda=use_cuda)
            model_trainer.init_model(create_model_function=
                                     (lambda model_name, problem: create_classifier_model(model_name,
                                                                                          problem,
                                                                                          encoded_size=args.encoded_size)))
            training_loop_method = model_trainer.train_autoencoder
            testing_loop_method = model_trainer.test_somatic_classifer
        else:
            model_trainer=None
            print("unknown mode specified: " + args.mode)
            exit(1)

        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)


        return model_trainer.training_loops(training_loop_method=training_loop_method,
                                            testing_loop_method=testing_loop_method)


    train_once(args, problem, use_cuda)
    exit(0)
