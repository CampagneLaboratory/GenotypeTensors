'''Train an auto-encoder for .vec/vecp files.'''
from __future__ import print_function

import argparse
import sys

# MIT License
#
# Copyright (c) 2018 Fabien Campagne
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
import torch

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer
from org.campagnelab.dl.problems.Problem import Problem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an auto-encoder for .vec files.')
    parser.add_argument('--supervised-model','-s', type=str, required=True,
                        help='Checkpoint key for the supervised model to adapt.')

    parser.add_argument('--adda-model', type=str,required=True,
                        help='Checkpoint key for the ADDA adaptation model.')
    parser.add_argument('--model-labels', help='Labels of the models to adapt. Coma separated list of labels',
                        type=str, default="best,latest")

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    problem = Problem()
    # Load the adaptation model:
    print("Loading adaptation model " + args.adda_model)

    trainer = CommonTrainer(args, problem, use_cuda)
    args.checkpoint_key=args.adda_model
    adapt_state = trainer.load_checkpoint_state(model_label="best")
    adapt_model=adapt_state["model"]

    for model_label in args.model_labels.split(","):
        args.checkpoint_key = args.supervised_model
        trainer=CommonTrainer(args,problem,use_cuda)
        print("Loading supervised model " + args.supervised_model + " ({})".format(model_label))
        # Convert best model:
        state=trainer.load_checkpoint_state(model_label=model_label)

        epoch=state["epoch"]
        supervised_model=state["model"]
        test_loss = state["best_test_loss"]
        adapt_model.install_supervised_model(supervised_model)

        trainer.save_model(best_test_loss=test_loss,epoch=epoch, model=adapt_model,model_label=model_label+"-ADDA")
        print("Model adapted and saved with label "+model_label+"-ADDA")

