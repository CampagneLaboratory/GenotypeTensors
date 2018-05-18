'''Load a CUDA model and save it back for the CPU.'''
from __future__ import print_function

import argparse

import os
import torch as torch

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
from org.campagnelab.dl.problems.Problem import Problem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load a CUDA model and save it back for the CPU.')

    parser.add_argument('--checkpoint-key', help='key to load and save the checkpoint model.', type=str)
    parser.add_argument('--model-labels', help='Labels of the models to convert. Coma separated list of labels', type=str, default="best,latest")
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


    problem = Problem()
    device = torch.device("cpu")
    for model_label in args.model_labels.split(","):
        trainer=CommonTrainer(args,problem,use_cuda)
        # Convert best model:
        state=trainer.load_checkpoint_state(model_label=model_label)

        epoch=state["epoch"]
        model=state["model"]
        test_loss = state["best_test_loss"]
        model.to(device)
        trainer.net=model

        trainer.save_model(best_test_loss=test_loss,epoch=epoch, model=model,model_label=model_label)

    print("Model converted to CPU.")