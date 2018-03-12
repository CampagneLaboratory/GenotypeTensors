import argparse

import sys

import pickle
import threading

import torch
from pickle import dump

from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate normalization statistics for datasets of the genotyping problems.')
    parser.add_argument('--vector-name', default="input", type=str,help="Name of the vector to estimate normalization statistics for.")
    parser.add_argument('--problem', default="genotyping:basename", type=str,
                        help='The problem dataset name. basename is used to locate file named '
                             'basename-train.vec, basename-validation.vec, basename-test.vec, basename-unlabeled.vec')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=512)
    parser.add_argument("--num-workers", type=int, default=0, help='Number of workers to feed data to the GPUs.')
    parser.add_argument("-n", type=int, default=sys.maxsize, help='Maximum number of examples to sample from each dataset.')

    args = parser.parse_args()
    problem = None
    if args.problem.startswith("genotyping:"):
        problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    elif args.problem.startswith("somatic:"):
        problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem, num_workers=args.num_workers)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)

    mean = None
    std = None
    n=0

    lock=threading.Lock()

    def normalize_mean_std(x):
        global mean
        global std
        global n
        with lock:
            if mean is None:
                # Use the first batch to estimate mean and std:
                mean = torch.mean(x, dim=0)
                # TODO: this is the wrong way to estimate std, need to estimate over entire datasets in another passe
                # TODO: after mean is known.
                std = torch.std(x, dim=0)
            else:
                mean += torch.mean(x, dim=0)
                std += torch.std(x, dim=0)

            n=n+1
        return x


    datasets = [problem.train_set(), problem.validation_set(), problem.unlabeled_set(), problem.test_set()]
    for index, dataset in enumerate(datasets):
        train_loader_subset = problem.loader_for_dataset(dataset,shuffle=True)
        print("Processing dataset {}/{}".format(index+1,len(datasets)))
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset),
            is_cuda=False,
            batch_names=["dataset"],
            volatile={"dataset": [args.vector_name]},
            recode_functions={
                args.vector_name: normalize_mean_std
            })
        batch_index=0
        for example in data_provider:
            batch_index+=1
            if batch_index*args.mini_batch_size>args.n:
                break
        data_provider.close()
    mean_global=mean/n
    std_global=std/n
    with open(problem.basename+"_"+args.vector_name+".mean",mode="wb") as mean_file:
        dump(mean_global,file=mean_file)

    with open(problem.basename+"_"+args.vector_name+".std",mode="wb") as std_file:
       dump(std_global,file=std_file)
    print("mean and std estimated and written.")
    sys.exit(0)