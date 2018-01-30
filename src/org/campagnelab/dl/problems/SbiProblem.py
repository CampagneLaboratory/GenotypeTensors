from pathlib import Path

import torch

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import GenotypeDataset, EmptyDataset, \
    InterleaveDatasets
from org.campagnelab.dl.problems.Problem import Problem




class SbiProblem(Problem):
    def get_vector_names(self):
        return []

    def input_size(self, input_name):
        return self.meta_data.get_vector_dimensions_from_name(input_name)

    def output_size(self, output_name):
        return self.meta_data.get_vector_dimensions_from_name(output_name)

    def train_set(self):
        return GenotypeDataset(self.basename + "-train.vec", vector_names=self.get_vector_names())

    def unlabeled_set(self):

        if self.file_exists(self.basename + "-unlabeled.list"):
            # Use a list of datasets and interleave their records:
            with open(self.basename + "-unlabeled.list") as list_file:
                lines=list_file.readlines()
                return InterleaveDatasets([ GenotypeDataset(path,vector_names=self.get_vector_names()) for path in lines ])
        else:
            if self.file_exists(self.basename + "-unlabeled.vec"):
                return GenotypeDataset(self.basename + "-unlabeled.vec", vector_names=self.get_vector_names())
            else:
                return EmptyDataset()

    def validation_set(self):
        if self.file_exists(self.basename + "-validation.vec"):
            return GenotypeDataset(self.basename + "-validation.vec",  vector_names=self.get_vector_names())
        else:
            return EmptyDataset()

    def __init__(self, mini_batch_size, code, num_workers=0):
        super().__init__(mini_batch_size)
        self.basename = code[len(self.basename_prefix()):]
        self.num_workers = num_workers
        self.meta_data = VectorReader(self.basename + "-train.vec", False, vector_names=[]).vector_reader_properties

    def train_loader(self):
        """Returns the torch dataloader over the training set. """

        return self.train_set().batch(batchsize=self.mini_batch_size(), policy='skip-last')

    def _filter(self, indices, iterator):
        fast_indices = set(indices)
        for (i, x) in enumerate(iterator, self.mini_batch_size()):
            if i in fast_indices:
                yield x

    def train_loader_subset(self, indices):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        assert False, "Not support for text .vec files"

    def validation_loader(self):
        """Returns the torch dataloader over the test set. """
        return self.validation_set().batch(batchsize=self.mini_batch_size(), policy='skip-last')

    def test_loader_subset(self, indices):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        assert False, "Not support for text .vec files"

    def unlabeled_loader(self):
        return self.unlabeled_set().batch(batchsize=self.mini_batch_size(), policy='skip-last')

    def reg_loader_subset(self, indices):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        # transform the unsupervised set the same way as the training set:

        assert False, "Not support for text .vec files"

    def loader_for_dataset(self, dataset):
        return dataset.batch(batchsize=self.mini_batch_size(), policy='skip-last')

    def loss_function(self, output_name):
        return torch.nn.CrossEntropyLoss()

    def file_exists(self, filename):
        return Path(filename).is_file()

    def basename_prefix(self):
        return None

class SbiSomaticProblem(SbiProblem):
    def name(self):
        return self.basename_prefix() + self.basename

    def get_vector_names(self):
        return ["input", "isBaseMutated", "somaticFrequency"]

    def basename_prefix(self):
        return "somatic:"


class SbiGenotypingProblem(SbiProblem):
    """A problem that exposes a set of sbi files converted to .vec/.vecp format. Files must
    be named using the convention: basename-train.vec, basename-validation.vec basename-unlabeled.vec
    and have associated .vecp meta-data files.
    """

    def name(self):
        return self.basename_prefix() + self.basename

    def basename_prefix(self):
        return "genotyping:"

    def get_vector_names(self):
        return ["input", "softmaxGenotype"]


