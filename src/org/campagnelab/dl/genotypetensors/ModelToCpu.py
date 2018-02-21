'''Load a CUDA model and save it back for the CPU.'''
from __future__ import print_function

import argparse

import os
import torch as torch
from org.campagnelab.dl.pytorch.images.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.images.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.images.TrainModelSplit import TrainModelSplit
from org.campagnelab.dl.pytorch.images.models import *

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
from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer
from org.campagnelab.dl.problems.SbiProblem import SbiGenotypingProblem, SbiSomaticProblem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load a CUDA model and save it back for the CPU.')

    parser.add_argument('--checkpoint-key', help='key to load and save the checkpoint model.', type=str)
    parser.add_argument('--problem', default="genotyping:", type=str,
                        help='The problem, either genotpying: or somatic:')
    args = parser.parse_args()
    if args.checkpoint_key is None:
        print("You must specify a checkpoint key.")
        exit(1)

    print("Loading pre-trained model " + args.checkpoint_key)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("With CUDA")
    else:
        print("With CPU")


    problem = None
    if args.problem.startswith("genotyping:"):
        problem = SbiGenotypingProblem(args.mini_batch_size, code=args.problem)
    elif args.problem.startswith("somatic:"):
        problem = SbiSomaticProblem(args.mini_batch_size, code=args.problem)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)

    print('==> Loading model from checkpoint..')
    model = None
    assert os.path.isdir('{}/models/'.format(args.model_path)), 'Error: no models directory found!'
    checkpoint = None
    try:

        checkpoint_filename = '{}/models/pytorch_{}_{}.t7'.format(args.model_path, args.checkpoint_key,
                                                                  args.model_label)
        checkpoint = torch.load(checkpoint_filename)
    except                FileNotFoundError:
        print("Unable to load model {} from checkpoint".format(args.checkpoint_key))
        exit(1)

    if checkpoint is not None:
        model = checkpoint['model']

    trainer=CommonTrainer(args,problem,use_cuda)
    model=trainer.load_checkpoint()
    model.cpu()
    trainer.net=model
    trainer.save_checkpoint(model)
    model_trainer.save_pretrained_model()
    print("Model converted to CPU.")