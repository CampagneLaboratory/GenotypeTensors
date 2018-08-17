'''Train an auto-encoder for .vec/vecp files.'''
from __future__ import print_function

import argparse
import collections
import concurrent
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import gc
from math import log, log2
from queue import Queue

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
from org.campagnelab.dl.genotypetensors.structured.Models import LoadedTensor
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider, DataProvider
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem
from org.campagnelab.dl.problems.StructuredSbiProblem import StructuredSbiGenotypingProblem
from org.campagnelab.dl.utils.utils import progress_bar
from torchfold import torchfold


def create_new_trainer(trainer_command_line):
    global trainers
    trainer_parser = define_train_auto_encoder_parser()
    trainer_args = trainer_parser.parse_args(trainer_command_line.split())
    if trainer_args.max_examples_per_epoch is None:
        trainer_args.max_examples_per_epoch = trainer_args.num_training
    trainer_args.num_training = args.num_training
    trainer_args.num_validation = args.num_validation
    trainer_args.no_progress = True
    trainer_args.epoch_min_accuracy = -1
    if args.lr is not None:
        trainer_args.lr = args.lr
    # print("Executing " + trainer_args.checkpoint_key)
    with open("args-{}".format(trainer_args.checkpoint_key), "w") as args_file:
        args_file.write(trainer_command_line + "--seed " + str(trainer_args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_parallel = False
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if trainer_args.max_examples_per_epoch is None:
        trainer_args.max_examples_per_epoch = args.num_training
    model_trainer, training_loop_method, testing_loop_method = configure_model_trainer(trainer_args,
                                                                                       problem,
                                                                                       device,
                                                                                       class_frequencies)
    model_trainer.set_common_lock(global_lock)
    model_trainer.args.max_epochs = args.max_epochs
    trainers += [model_trainer]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an auto-encoder for .vec files.')
    parser.add_argument('--commands', type=str, required=True,
                        help='Commands that start TrainAutoEncoder.py. One line per command.')
    parser.add_argument('--problem', default="genotyping:basename", type=str,
                        help='The genotyping problem dataset name. basename is used to locate file named '
                             'basename-train.vec, basename-validation.vec, basename-unlabeled.vec')
    parser.add_argument('--mode', help='Training mode: determines how the mini-batch is loaded. When supervised,'
                                       'only training set examples are loaded. ',
                        default="supervised_direct",
                        choices=["supervised_direct"])
    parser.add_argument('--num-training', '-n', "--max-training-examples", type=int,
                        help='Maximum number of training examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--num-validation', '-v', '-x', type=int, help='Maximum number of validation examples to use.',
                        default=sys.maxsize)
    parser.add_argument('--min-iterations', type=int,
                        help='Minimum number of minibatches to use for training the first iteration.',
                        default=1)
    parser.add_argument('--fixed-iterations', action='store_true', help='Never increment the number of iterations. Fixed at min-iterations.')

    parser.add_argument("--lr", type=float, default=None, help='Learning rate (overrides configurations)')
    parser.add_argument("--num-workers", type=int, default=0, help='Number of workers to feed data to the GPUs.')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=32)
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs to train for.', default=10)
    parser.add_argument('--max-models', type=int, help='Maximum number of models to train in one shot.',
                        default=sys.maxsize)
    parser.add_argument('--num-models-per-gpu', type=int, help='Maximum number of models to train on one GPU.',
                        default=0)
    parser.add_argument('--num-gpus', type=int, help='Number of GPUs to use for search.',
                        default=1)
    parser.add_argument('--num-estimate-class-frequencies', type=int, help='Number of examples to look at to estimate '
                                                                           'class frequencies.',
                        default=sys.maxsize)
    parser.add_argument('--debug', action='store_true', help='Used to debug some code.')
    parser.add_argument('--num-debug', type=int,
                        help='Maximum number of debug iterations.',
                        default=sys.maxsize)

    args = parser.parse_args()

    problem = None
    if args.problem.startswith("struct_genotyping:"):
        problem = StructuredSbiGenotypingProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)
    args.num_training = min(args.num_training, len(problem.train_set()))
    args.num_estimate_class_frequencies = min(args.num_estimate_class_frequencies, len(problem.train_set()))
    args.num_validation = min(args.num_validation, len(problem.validation_set()))
    trainers = []
    count = 0
    with open(args.commands, "r") as command_file:

        trainer_arguments = command_file.readlines()
        count = len(trainer_arguments)
        i = 0

    if args.num_models_per_gpu == 0:
        print("Running with {} models per gpu. Use --num-models-per-gpu to reduce concurrency if needed.".format(count))
        args.num_models_per_gpu = max(count, 1)

    # Estimate class frequencies:
    print("Estimating class frequencies..")
    train_loader_subset = problem.train_loader_subset_range(0, args.num_estimate_class_frequencies)
    class_frequencies = {}  # one frequency vector per output_name
    with DataProvider(iterator=zip(train_loader_subset), device=torch.device("cpu"),
                      batch_names=["training"], vectors_to_keep=problem.get_vector_names()) as data_provider:
        done = False
        for batch_idx, (_, data_dict) in enumerate(data_provider):
            if done:
                break
            for output_name in problem.get_output_names():
                target_s = data_dict["training"][output_name]
                if output_name not in class_frequencies.keys():
                    class_frequencies[output_name] = torch.ones(target_s.size(1))
                cf = class_frequencies[output_name]
                indices = torch.nonzero(target_s.data)
                indices = indices[:, 1]
                for index in range(indices.size(0)):
                    cf[indices[index]] += 1
                # for class_index in range(target_s.size(1)):
                #    cf[indices[:,1]] += 1
            if batch_idx * args.mini_batch_size > args.num_estimate_class_frequencies:
                done = True
            progress_bar(batch_idx * args.mini_batch_size,
                         args.num_estimate_class_frequencies,
                         "Class frequencies")
    print("class frequencies: " + str(class_frequencies))
    del train_loader_subset

    # Initialize the trainers:
    global_lock = threading.Lock()
    for trainer_command_line in trainer_arguments[0:args.max_models]:

        create_new_trainer(trainer_command_line)
        print("Configured {}/{} trainers".format(len(trainers), count))
        if (len(trainers) > args.max_models): break
    model_trainer_arg_index = args.max_models
    print("Executing hyper-parameter search for {} models.".format(len(trainers)))
    for model_trainer in trainers:
        model_trainer.class_frequency(class_frequencies=class_frequencies)


    def raise_to_do_exceptions(futures):
        # Report any exceptions encountered in to_do:
        for future in futures:
            try:
                future.result()
            except Exception as e:
                raise e


    cuda = torch.device('cuda')
    cpu = torch.device('cpu')
    if args.debug:
        with ThreadPoolExecutor(max_workers=args.num_models_per_gpu * args.num_gpus) as thread_executor:

            def todo(model_trainer):
                model_trainer.args.label_strategy = "UNIFORM"
                num_classes = problem.output_size("softmaxGenotype")[0]
                model_trainer.num_classes = num_classes
                model_trainer.epsilon_label_smoothing = 0.1

                return model_trainer.dreamup_target_for(category_prior=None,
                                                        num_classes=num_classes, input=None)


            for index in range(0, args.num_debug):
                futures = []
                for model_trainer in trainers:
                    futures += [thread_executor.submit(todo, model_trainer)]
                concurrent.futures.wait(futures)
                raise_to_do_exceptions(futures)
                progress_bar(index,
                             args.num_debug,
                             "Debug iterations")
        exit(0)


    def do_training_evaluate(thread_executor, step, data_provider, num_iterations, all_iterations):

        global trainers
        for model_trainer in trainers:
            model_trainer.training_performance_estimators.init_performance_metrics()
            model_trainer.test_performance_estimators.init_performance_metrics()

        todo_arguments = []
        num_batches = 0
        training_validation_batches = []
        for batch_idx, (_, data_dict) in enumerate(data_provider):
            sbi = data_dict["training"]["sbi"]
            preloaded_sbi_tensors = trainers[0].net.sbi_mapper.preload(sbi)
            target_s = data_dict["training"]["softmaxGenotype"]
            metadata = data_dict["training"]["metaData"]
            training_validation_batches.append([preloaded_sbi_tensors, target_s, metadata])

            if len(training_validation_batches) == 2:
                training_batch = training_validation_batches[0]
                validation_batch = training_validation_batches[1]
                # move forward by one slot so that validation becomes training:
                training_validation_batches = training_validation_batches[1:]

                num_batches += 1
                futures = []
                for model_trainer in trainers:
                    def to_do(model_trainer, batch_idx, training_batch, validation_batch):
                        preloaded_sbi_tensors, target_s, metadata = training_batch

                        prepare_batch(model_trainer, preloaded_sbi_tensors, target_s, metadata)
                        model_trainer.net.train()
                        model_trainer.train_one_batch(model_trainer.training_performance_estimators,
                                                      batch_idx, None, None, None)
                        model_trainer.net.eval()
                        preloaded_sbi_tensors, target_s, metadata = validation_batch
                        prepare_batch(model_trainer, preloaded_sbi_tensors, target_s, metadata)
                        model_trainer.test_one_batch(model_trainer.test_performance_estimators,
                                                     batch_idx, sbi=None, target_s=None, errors=None)
                        del model_trainer.batch

                    futures += [
                        thread_executor.submit(to_do, model_trainer, batch_idx, training_batch, validation_batch)]

                concurrent.futures.wait(futures)
                # progress_bar(batch_idx * args.mini_batch_size,
                #             args.num_training,
                #             msg="Training phase")
                # Report any exceptions encountered in to_do:
                raise_to_do_exceptions(futures)
                if batch_idx > num_iterations:
                    break
        performance = []
        for model_trainer in trainers:
            perfs = PerformanceList()
            perfs += model_trainer.test_performance_estimators
            performance.append({'test_loss': perfs.get_metric("test_supervised_loss"), "trainer": model_trainer})
        all_trainers = sorted(performance, key=lambda x: x['test_loss'])
        print('After {} iterations, best score: {} worse score: {}, {} models created'.format(all_iterations,
                                                                                              all_trainers[0][
                                                                                                  'test_loss'],
                                                                                              all_trainers[-1][
                                                                                                  'test_loss'],
                                                                                              model_trainer_arg_index))
        trimed_length = (len(all_trainers) / 2)
        if trimed_length >= 1:
            best_trainers = all_trainers[0:int(trimed_length)]
            trainers = []
            for model_trainer in best_trainers:
                trainers.append(model_trainer['trainer'])
        # log performance of best models seen so far:
        for model_trainer in [t['trainer'] for t in reversed(all_trainers)]:
            perfs = PerformanceList()
            perfs += model_trainer.training_performance_estimators
            perfs += model_trainer.test_performance_estimators
            if all_iterations == 1:
                model_trainer.log_performance_header(perfs)

            early_stop, perfs = model_trainer.log_performance_metrics(step, perfs)


    def prepare_batch(model_trainer, preloaded_sbi_tensors, target_s, metadata):
        model_trainer.batch = []
        model_trainer.fold = torchfold.Fold(model_trainer.fold_executor)
        # preloaded_sbi_tensors = model_trainer.net.sbi_mapper.preload(sbi)

        for example_index in range(target_s.size(0)):
            model_trainer.batch.append((preloaded_sbi_tensors[example_index].clone(device=cuda),
                                        LoadedTensor(target_s[example_index].view(1, -1)).clone(device=cuda),
                                        LoadedTensor(metadata[example_index].view(1, -1)).clone(device=cuda)))


    with  ThreadPoolExecutor(max_workers=args.num_models_per_gpu * args.num_gpus) as thread_executor:
        train_loader_subset = problem.train_loader_subset_range(0, args.num_training)
        data_provider = DataProvider(
            iterator=zip(train_loader_subset),
            device=cpu,
            batch_names=["training"],
            requires_grad={"training": ["sbi"]},
            vectors_to_keep=["metaData", "softmaxGenotype"]
        )
        try:
            num_iterations = args.min_iterations
            all_iterations = 1
            while all_iterations < len(train_loader_subset):
                # print("Training {} workers for {} iterations".format(len(trainers), num_iterations))
                all_iterations += num_iterations
                do_training_evaluate(thread_executor, all_iterations, data_provider, num_iterations, all_iterations)
                gc.collect()
                if not args.fixed_iterations:
                    num_iterations += 1

                while len(trainers) < args.max_models:
                    create_new_trainer(trainer_arguments[model_trainer_arg_index])
                    model_trainer_arg_index += 1
        finally:
            data_provider.close()
    # don't wait for threads to die, just exit:
    os._exit(0)