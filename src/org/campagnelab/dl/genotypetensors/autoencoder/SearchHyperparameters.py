'''Train an auto-encoder for .vec/vecp files.'''
from __future__ import print_function

import argparse
import concurrent
import os
import sys
from concurrent.futures import ThreadPoolExecutor

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
from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import recode_for_label_smoothing
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider, DataProvider
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem
from org.campagnelab.dl.utils.utils import progress_bar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an auto-encoder for .vec files.')
    parser.add_argument('--commands', type=str, required=True,
                        help='Commands that start TrainAutoEncoder.py. One line per command.')
    parser.add_argument('--problem', default="genotyping:basename", type=str,
                        help='The genotyping problem dataset name. basename is used to locate file named '
                             'basename-train.vec, basename-validation.vec, basename-unlabeled.vec')
    parser.add_argument('--mode', help='Training mode: determines how the mini-batch is loaded. When supervised,'
                                       'only training set examples are loaded. With semi-supervised, both training '
                                       'set and unlabeled set are loaded.',
                        default="supervised",
                        choices=["supervised", "semi-supervised"])
    parser.add_argument('--num-training', '-n', "--max-training-examples", type=int,
                        help='Maximum number of training examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--num-validation', '-v', type=int, help='Maximum number of validation examples to use.',
                        default=sys.maxsize)
    parser.add_argument("--num-workers", type=int, default=0, help='Number of workers to feed data to the GPUs.')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=1000)
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs to train for.', default=10)
    parser.add_argument('--max-models', type=int, help='Maximum number of models to train in one shot.',
                        default=sys.maxsize)
    parser.add_argument('--num-models-per-gpu', type=int, help='Maximum number of models to train on one GPU.',
                            default=30)
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use for search.',
                        default=1)
    args = parser.parse_args()

    problem = None
    if args.problem.startswith("genotyping:"):
        problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    elif args.problem.startswith("somatic:"):
        problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)

    trainers = []
    # Initialize the trainers:
    with open(args.commands, "r") as command_file:

        trainer_arguments = command_file.readlines()
        count = len(trainer_arguments)
        i = 0


    problem = None
    if args.problem.startswith("genotyping:"):
        problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    elif args.problem.startswith("somatic:"):
        problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)

    # Estimate class frequencies:
    print("Estimating class frequencies..")
    train_loader_subset = problem.train_loader_subset_range(0, min(100000, args.num_training))
    data_provider = DataProvider(iterator=zip(train_loader_subset), is_cuda=False,
                                                    batch_names=["training"],
                                                    volatile={"training": problem.get_vector_names()},
                                                    )

    class_frequencies = {}  # one frequency vector per output_name
    done = False
    for batch_idx, (_, data_dict) in enumerate(data_provider):
        if done:
            break
        for output_name in problem.get_output_names():
            target_s = data_dict["training"][output_name]
            if output_name not in class_frequencies.keys():
                class_frequencies[output_name] = torch.ones(target_s[0].size())
            cf = class_frequencies[output_name]
            indices = torch.nonzero(target_s.data)
            for example_index in range(len(target_s)):
                cf[indices[example_index, 1]] += 1

        progress_bar(batch_idx * args.mini_batch_size,
                     args.num_training,
                     "Class frequencies")

        for trainer_command_line in trainer_arguments:
            trainer_parser = define_train_auto_encoder_parser()
            trainer_args = trainer_parser.parse_args(trainer_command_line.split())

            if trainer_args.max_examples_per_epoch is None:
                trainer_args.max_examples_per_epoch = trainer_args.num_training
            trainer_args.num_training = args.num_training
            print("Executing " + trainer_args.checkpoint_key)

            with open("args-{}".format(trainer_args.checkpoint_key), "w") as args_file:
                args_file.write(trainer_command_line + "--seed " + str(trainer_args.seed))

            use_cuda = torch.cuda.is_available()
            is_parallel = False
            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

            if trainer_args.max_examples_per_epoch is None:
                trainer_args.max_examples_per_epoch = args.num_training
            model_trainer, training_loop_method, testing_loop_method = configure_model_trainer(trainer_args,
                                                                                               problem,
                                                                                               use_cuda,
                                                                                               class_frequencies)
            trainers += [model_trainer]
            print("Configured {}/{} trainers".format(len(trainers), count))
            if (len(trainers) > args.max_models): break

    print("Executing hyper-parameter search for {} models.".format(len(trainers)))
    for model_trainer in trainers:
            model_trainer.class_frequency(class_frequencies=class_frequencies)

    if args.mode == "supervised":
        def do_training(epoch,thread_executor):
            print('Training, epoch: %d' % epoch)
            for model_trainer in trainers:
                model_trainer.training_performance_estimators.init_performance_metrics()

            unsupervised_loss_acc = 0
            num_batches = 0
            train_loader_subset = problem.train_loader_subset_range(0, args.num_training)
            data_provider = DataProvider(
                iterator=zip(train_loader_subset),
                is_cuda=use_cuda,
                batch_names=["training"],
                requires_grad={"training": ["input"]},
                volatile={"training": ["metaData"]},
                recode_functions={
                    "softmaxGenotype": lambda x: recode_for_label_smoothing(x, 0),
                }
            )
            try:

                snp_weight = 1.0
                for batch_idx, (_, data_dict) in enumerate(data_provider):
                    input_s = data_dict["training"]["input"]
                    target_s = data_dict["training"]["softmaxGenotype"]
                    metadata = data_dict["training"]["metaData"]
                    batch_size = len(input_s)
                    num_batches += 1
                    futures=[]
                    for model_trainer in trainers:
                        def to_do(model_trainer, input_s, target_s, metadata):
                            input_s_local = input_s.clone()
                            target_s_local = target_s.clone()
                            model_trainer.net.train()
                            target_s_smoothed = recode_for_label_smoothing(target_s_local,
                                                                  model_trainer.args.epsilon_label_smoothing)

                            model_trainer.train_one_batch(model_trainer.training_performance_estimators,
                                                          batch_idx, input_s_local, target_s_smoothed, metadata)
                        futures+=[thread_executor.submit(to_do,model_trainer,input_s,target_s,metadata)]
                    concurrent.futures.wait(futures)
                    if (batch_idx + 1) * batch_size > args.num_training:
                        break
            finally:
                data_provider.close()


        def do_testing(epoch, thread_executor):
            print('Testing, epoch: %d' % epoch)
            errors = None

            for model_trainer in trainers:
                model_trainer.test_performance_estimators.init_performance_metrics()

            validation_loader_subset = problem.validation_loader_range(0, args.num_validation)
            data_provider = DataProvider(
                iterator=zip(validation_loader_subset),
                is_cuda=use_cuda,
                batch_names=["validation"],
                requires_grad={"validation": []},
                volatile={
                    "validation": ["input", "softmaxGenotype"]
                },

            )
            try:
                for batch_idx, (_, data_dict) in enumerate(data_provider):
                    input_s = data_dict["validation"]["input"]
                    target_s = data_dict["validation"]["softmaxGenotype"]
                    futures=[]
                    for model_trainer in trainers:
                        def to_do(model_trainer,input_s,target_s):
                            input_s_local=input_s.clone()
                            target_s_local=target_s.clone()
                            target_smoothed = recode_for_label_smoothing(target_s_local,
                                                                  model_trainer.args.epsilon_label_smoothing)

                            model_trainer.net.eval()
                            model_trainer.test_one_batch(model_trainer.test_performance_estimators,
                                                         batch_idx, input_s_local, target_smoothed,errors)

                        futures+=[thread_executor.submit(to_do,model_trainer,input_s,target_s)]
                    concurrent.futures.wait(futures)
            finally:
                data_provider.close()
            #print("test errors by class: ", str(errors))

            for model_trainer in trainers:
                perfs = PerformanceList()
                perfs+=[model_trainer.training_performance_estimators]
                perfs+=[model_trainer.test_performance_estimators]
                if epoch==0:

                    model_trainer.log_performance_header(perfs)

                early_stop, perfs = model_trainer.log_performance_metrics(epoch, perfs)

                if early_stop:
                    # early stopping requested, no longer train this model:
                    trainers.remove(model_trainer)
                    return
                # Apply learning rate schedule:
                test_metric = model_trainer.test_performance_estimators.get_metric(model_trainer.get_test_metric_name())
                assert test_metric is not None, (model_trainer.get_test_metric_name() +
                                                 "must be found among estimated performance metrics")
                if not model_trainer.args.constant_learning_rates:
                    model_trainer.scheduler_train.step(test_metric, epoch)


    else:
        pass
    with  ThreadPoolExecutor(max_workers=args.num_models_per_gpu*args.num_gpus) as thread_executor:
        for epoch in range(args.max_epochs):
            do_training(epoch,thread_executor)
            do_testing(epoch,thread_executor)
            if len(trainers) == 0:
                break

    # don't wait for threads to die, just exit:
    os._exit(0)
