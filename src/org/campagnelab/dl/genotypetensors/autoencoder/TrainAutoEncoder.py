'''Train an auto-encoder for .vec/vecp files.'''
from __future__ import print_function

import os
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
from org.campagnelab.dl.genotypetensors.autoencoder.ModelTrainers import configure_model_trainer, \
    define_train_auto_encoder_parser
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem

if __name__ == '__main__':

    parser=define_train_auto_encoder_parser()
    args = parser.parse_args()

    if args.max_examples_per_epoch is None:
        args.max_examples_per_epoch = args.num_training

    print("Executing " + args.checkpoint_key)

    with open("args-{}".format(args.checkpoint_key), "w") as args_file:
        args_file.write(" ".join(sys.argv + ["--seed ", str(args.seed)]))

    use_cuda = torch.cuda.is_available()
    is_parallel = False
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    problem = None
    if args.problem.startswith("genotyping:"):
        problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    elif args.problem.startswith("somatic:"):
        problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)


    def get_metric_value(all_perfs, query_metric_name):
        for perf in all_perfs:
            metric = perf.get_metric(query_metric_name)
            if metric is not None:
                return metric


    def train_once(train_args, train_problem, train_use_cuda):
        train_problem.describe()
        training_loop_method = None
        testing_loop_method = None
        model_trainer, training_loop_method, testing_loop_method=configure_model_trainer(train_args,train_problem,train_use_cuda)

        torch.manual_seed(train_args.seed)
        if train_use_cuda:
            torch.cuda.manual_seed(train_args.seed)

        return model_trainer.training_loops(training_loop_method=training_loop_method,
                                            testing_loop_method=testing_loop_method)


    train_once(args, problem, use_cuda)
    # don't wait for threads to die, just exit:
    os._exit(0)
