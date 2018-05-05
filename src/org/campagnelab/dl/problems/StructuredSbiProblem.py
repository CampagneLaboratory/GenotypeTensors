from pathlib import Path

import copy
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import GenotypeDataset, EmptyDataset, \
    InterleaveDatasets, CyclicInterleavedDatasets, CachedGenotypeDataset, DispatchDataset
from org.campagnelab.dl.genotypetensors.structured.Datasets import StructuredGenotypeDataset
from org.campagnelab.dl.problems.Problem import Problem
from org.campagnelab.dl.problems.SbiProblem import SbiProblem


def collate_sbi(batch):
    """ For each batch, organize sbi messages in a list, returned as first element of the batch tuple"""
    element = batch[0]

    if isinstance(element[0],dict)  and element[0]['type']=="BaseInformation":
        # note that batch_element[1][1] drops the index of the example: batch_element[1][0] to
        # keep only softmaxGenotype and metaData.
        return {'sbi':[batch_element[0] for batch_element in batch]}, \
               default_collate([ batch_element[1][1] for batch_element in batch])
    else:
        return default_collate(batch)

class StructuredSbiGenotypingProblem(SbiProblem):
    """An SBI problem where the tensors are generated from structured messages directly from an SBI, and
    labels are loaded from the vec file. """
    # TODO: overwrite the dataset and loader methods to use StructuredGenotypeDataset
    def name(self):
        return self.basename_prefix() + self.basename

    def basename_prefix(self):
        return "struct_genotyping:"

    def get_input_names(self):
        return []

    def get_vector_names(self):
        return [ "softmaxGenotype","metaData"]

    def train_set(self):
        return StructuredGenotypeDataset(self.basename+"-train")

    def validation_set(self):
        return StructuredGenotypeDataset(self.basename + "-validation")

    def test_set(self):
        return StructuredGenotypeDataset(self.basename + "-test")

    def unlabeled_set(self):
        return StructuredGenotypeDataset(self.basename + "-unlabeled")

    def loader_for_dataset(self, dataset, shuffle=False):
        return iter(DataLoader(dataset=dataset, shuffle=False, batch_size=self.mini_batch_size(),collate_fn=lambda batch:collate_sbi(batch),
                               num_workers=self.num_workers, pin_memory=False, drop_last=self.drop_last_batch))


    def get_output_names(self):
        return ["softmaxGenotype"]
