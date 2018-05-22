'''Estimate performance on the test set.'''
from __future__ import print_function

import argparse
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
from org.campagnelab.dl.genotypetensors.autoencoder.PredictModel import PredictModel
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem
from org.campagnelab.dl.problems.StructuredSbiProblem import StructuredSbiGenotypingProblem


def format_nice(n):
    try:
        if n == int(n):
            return str(n)
        if n == float(n):
            if n<0.001:
                return "{0:.3E}".format(n)
            else:
                return "{0:.4f}".format(n)
    except:
        return str(n)


parser = argparse.ArgumentParser(description='Save predicted output of a model on a dataset.')
parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=128)
parser.add_argument('-n', type=int, help='number of examples to predict.', default=sys.maxsize)
parser.add_argument('--checkpoint-key', help='Random key to load a checkpoint model',
                    default=None)
parser.add_argument('--model-path', help='Path of where the models directory is located.',      default=None)
parser.add_argument('--model-label', help='Model label: best or latest.',      default="best")
parser.add_argument('-o','--output', help='Output file name.',      default="output")
parser.add_argument('--problem', default="genotyping:basename", type=str,
                    help='The problem, genotyping:basename or somatic:basename')
parser.add_argument('--dataset', default="validation", type=str,
                    help='the dataset to predict on. Filename used will be basename-dataset, where basename if given in the problem.')
parser.add_argument('--num-workers', default=0, type=int, help="Number of workers to use when parallel or partitioned "
                                                               "dataset is used")
parser.add_argument("--processing", choices=["multithreaded", "sequential", "parallel", "partitioned"], type=str,
                    default="multithreaded", help="Type of processing to use")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('==> Loading model from checkpoint..')
model = None
assert os.path.isdir('{}/models/'.format(args.model_path)), 'Error: no models directory found!'
checkpoint = None
try:

    checkpoint_filename='{}/models/pytorch_{}_{}.t7'.format(args.model_path,args.checkpoint_key, args.model_label)
    checkpoint = torch.load(checkpoint_filename)
except                FileNotFoundError:
    print("Unable to load model {} from checkpoint".format(args.checkpoint_key))
    exit(1)

if checkpoint is not None:
    model = checkpoint['model']
problem = None
if args.problem.startswith("genotyping:"):
    problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem, drop_last_batch=False,
                                   num_workers=args.num_workers)
elif args.problem.startswith("somatic:"):
    problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem, drop_last_batch=False,
                                num_workers=args.num_workers)
elif args.problem.startswith("struct_genotyping:"):
    problem = StructuredSbiGenotypingProblem(args.mini_batch_size, code=args.problem, drop_last_batch=False,
                                             num_workers=args.num_workers)
else:
    print("Unsupported problem: " + args.problem)
    exit(1)

if problem is None or model is None:
    print("no problem or model, aborting")
    exit(1)

domain_descriptor = checkpoint["domain_descriptor"] if "domain_descriptor" in checkpoint else None
feature_mapper = checkpoint["feature_mapper"] if "feature_mapper" in checkpoint else None
samples = checkpoint["samples"] if "samples" in checkpoint else None
input_files = checkpoint["input_files"] if "input_files" in checkpoint else None


normalize=hasattr(model,"prenormalized_inputs") and model.prenormalized_inputs

tester = PredictModel(model=model, problem=problem, device=device,
                      domain_descriptor=domain_descriptor,
                      feature_mapper=feature_mapper, samples=samples, input_files=input_files,
                      processing_type=args.processing, num_workers=args.num_workers,
                      normalize=normalize)

iterator=None
if args.dataset=="validation":
    args.n = min(args.n, len(problem.validation_set()))
    iterator=problem.validation_loader_range(0, args.n)
elif args.dataset=="unlabeled":
    args.n = min(args.n, len(problem.unlabeled_set()))
    iterator = problem.unlabeled_loader_range(0, args.n)
elif args.dataset=="test":
    args.n = min(args.n, len(problem.test_set()))
    iterator = problem.test_loader_range(0, args.n)
elif args.dataset == "train":
    args.n = min(args.n, len(problem.train_set()))
    iterator = problem.train_loader_subset_range(0, args.n)

else:
    print("Unsupported dataset: " + args.dataset)
    exit(1)

tester.predict(iterator=iterator, output_filename=args.output, max_examples=args.n)

# don't wait for threads to die, just exit:
os._exit(0)