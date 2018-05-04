from pathlib import Path

import copy
import torch
from torch.utils.data import DataLoader, ConcatDataset

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import GenotypeDataset, EmptyDataset, \
    InterleaveDatasets, CyclicInterleavedDatasets, CachedGenotypeDataset, DispatchDataset
from org.campagnelab.dl.problems.Problem import Problem
from org.campagnelab.dl.problems.SbiProblem import SbiProblem


class StructuredSbiGenotypingProblem(SbiProblem):
    """An SBI problem where the tensors are generated from structured messages directly from an SBI, and
    labels are loaded from the vec file. """
    # TODO: overwrite the dataset and loader methods to use StructuredGenotypeDataset
    def name(self):
        return self.basename_prefix() + self.basename

    def basename_prefix(self):
        return "struct_genotyping:"

    def get_input_names(self):
        return ["sbi"]

    def get_vector_names(self):
        return ["sbi", "softmaxGenotype","metaData"]

    def get_output_names(self):
        return ["softmaxGenotype"]
