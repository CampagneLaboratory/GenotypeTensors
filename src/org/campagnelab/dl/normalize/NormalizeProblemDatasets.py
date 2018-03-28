import argparse

import sys

import threading

import torch

from org.campagnelab.dl.multithreading.sequential_implementation import DataProvider
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate normalization statistics for datasets of the genotyping '
                                                 'problems.')
    parser.add_argument('--vector-name', default="input", type=str, help="Name of the vector to estimate normalization "
                                                                         "statistics for.")
    parser.add_argument('--problem', default="genotyping:basename", type=str,
                        help='The problem dataset name. basename is used to locate file named '
                             'basename-train.vec, basename-validation.vec, basename-test.vec, basename-unlabeled.vec')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=512)
    parser.add_argument("--num-workers", type=int, default=0, help='Number of workers to feed data to the GPUs.')
    parser.add_argument("-n", type=int, default=sys.maxsize, help='Maximum number of examples to sample from each '
                                                                  'dataset.')

    args = parser.parse_args()
    problem = None
    if args.problem.startswith("genotyping:"):
        problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    elif args.problem.startswith("somatic:"):
        problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)

    sum_n = None
    n = 0
    mean = None
    sum_sdm_n = None
    std = None

    lock = threading.Lock()

    def add_to_sum(x):
        # Add each row of minibatch x to global sum sum_n
        global sum_n
        global n
        with lock:
            if sum_n is None:
                sum_n = torch.sum(x, dim=0)
            else:
                sum_n += torch.sum(x, dim=0)
            n += x.size()[0]

    def add_to_sum_sdm(x):
        # Add squared difference between each row of minibatch x and mean to global sum sum_sdm_n
        global sum_sdm_n
        global mean
        with lock:
            if sum_sdm_n is None:
                sum_sdm_n = torch.sum(torch.pow(x - mean.expand_as(x), 2), dim=0)
            else:
                sum_sdm_n += torch.sum(torch.pow(x - mean.expand_as(x), 2), dim=0)


    datasets = [problem.train_set(), problem.validation_set(), problem.unlabeled_set(), problem.test_set()]
    # First, get the sum of all of the example vectors
    for index, dataset in enumerate(datasets):
        train_loader_subset = problem.loader_for_dataset(dataset, shuffle=True)
        print("Summing dataset {}/{}".format(index + 1, len(datasets)))
        data_provider = DataProvider(
            iterator=zip(train_loader_subset),
            is_cuda=False,
            batch_names=["dataset"],
            volatile={"dataset": [args.vector_name]},
            recode_functions={
                args.vector_name: add_to_sum
            })
        batch_index = 0
        for example in data_provider:
            batch_index += 1
            if batch_index * args.mini_batch_size > args.n:
                break
        data_provider.close()
    # Calculate the mean
    print("Calculating the mean")
    mean = sum_n / n
    # Calculate the sum of squared deviations from means (sum_sdm)
    for index, dataset in enumerate(datasets):
        train_loader_subset = problem.loader_for_dataset(dataset, shuffle=True)
        print("Calculating sum of squared deviations for dataset {}/{}".format(index + 1, len(datasets)))
        data_provider = DataProvider(
            iterator=zip(train_loader_subset),
            is_cuda=False,
            batch_names=["dataset"],
            volatile={"dataset": [args.vector_name]},
            recode_functions={
                args.vector_name: add_to_sum_sdm
            })
        batch_index = 0
        for example in data_provider:
            batch_index += 1
            if batch_index * args.mini_batch_size > args.n:
                break
        data_provider.close()
    # Calculate standard deviation as sqrt((sum_sdm * (1 / n - 1))
    print("Calculating the standard deviation")
    std = torch.pow(sum_sdm_n * (1 / (n - 1)), 0.5)
    torch.save(mean, problem.basename + "_" + args.vector_name + ".mean")
    torch.save(std, problem.basename + "_" + args.vector_name + ".std")
    print("Mean and std estimated and written.")
    sys.exit(0)
