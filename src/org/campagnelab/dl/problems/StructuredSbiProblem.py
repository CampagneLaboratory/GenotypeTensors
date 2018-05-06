import sys
from pathlib import Path

import copy
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torchnet.dataset.dataset import Dataset

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import GenotypeDataset, EmptyDataset, \
    InterleaveDatasets, CyclicInterleavedDatasets, CachedGenotypeDataset, DispatchDataset
from org.campagnelab.dl.genotypetensors.structured.Datasets import StructuredGenotypeDataset
from org.campagnelab.dl.problems.Problem import Problem
from org.campagnelab.dl.problems.SbiProblem import SbiProblem


def collate_sbi(batch):
    """ For each batch, organize sbi messages in a list, returned as first element of the batch tuple"""
    element = batch[0]

    if isinstance(element[0], dict) and element[0]['type'] == "BaseInformation":
        # note that batch_element[1][1] drops the index of the example: batch_element[1][0] to
        # keep only softmaxGenotype and metaData.
        example_indices = default_collate([batch_element[1][0] for batch_element in batch])
        data_map = {"sbi":[batch_element[0] for batch_element in batch]}
        data_map.update(   default_collate([batch_element[1][1] for batch_element in batch]))


        return example_indices, data_map
    else:
        return default_collate(batch)

class StructSmallerDataset(Dataset):
    def __init__(self, delegate, new_size):
        super().__init__()
        err = "new size must be smaller than or equal to delegate size."
        assert new_size <= len(delegate) or new_size == sys.maxsize, err
        self.length = new_size
        self.delegate = delegate

    def __len__(self):
        return min(self.length, len(self.delegate))

    def __getitem__(self, idx):
        if idx < self.length:
            value = self.delegate[idx]
            if idx==self.length-1:
                print("Closing Struct dataset.")
                self.delegate.close()
            return value
        else:
            assert False, "index is larger than trimmed length."


class StructuredSbiGenotypingProblem(SbiProblem):
    """An SBI problem where the tensors are generated from structured messages directly from an SBI, and
    labels are loaded from the vec file. """

    def name(self):
        return self.basename_prefix() + self.basename

    def basename_prefix(self):
        return "struct_genotyping:"

    def get_input_names(self):
        return []

    def get_vector_names(self):
        return ["softmaxGenotype", "metaData"]

    def train_set(self):
        return StructuredGenotypeDataset(self.basename + "-train")

    def validation_set(self):
        return StructuredGenotypeDataset(self.basename + "-validation")

    def test_set(self):
        return StructuredGenotypeDataset(self.basename + "-test")

    def unlabeled_set(self):
        if self.file_exists(self.basename + "-unlabeled.list"):
            return StructuredGenotypeDataset(self.basename + "-unlabeled")
        else:
            return EmptyDataset()

    def loader_for_dataset(self, dataset, shuffle=False):
        return iter(DataLoader(dataset=dataset, shuffle=False, batch_size=self.mini_batch_size(),
                               collate_fn=lambda batch: collate_sbi(batch),
                               num_workers=0, pin_memory=False, drop_last=self.drop_last_batch))
    def loader_subset_range(self,dataset, start, end):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        if start==0:
            return self.loader_for_dataset(StructSmallerDataset(delegate=dataset, new_size=end),shuffle=False)
        else:
            return self.train_loader_subset(range(start, end))

    def train_loader_subset_range(self, start, end):
        return self.loader_subset_range(self.train_set(),start,end)

    def validation_loader_subset_range(self, start, end):
        return self.loader_subset_range(self.validation_set(),start,end)

    def test_loader_subset_range(self, start, end):
        return self.loader_subset_range(self.test_set(),start,end)

    def unlabeled_loader_subset_range(self, start, end):
        return self.loader_subset_range(self.unlabeled_set(),start,end)

    def get_output_names(self):
        return ["softmaxGenotype"]
