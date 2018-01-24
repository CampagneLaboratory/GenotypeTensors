from pathlib import Path

import torch

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import GenotypeDataset, EmptyDataset
from org.campagnelab.dl.problems.Problem import Problem


class SbiGenotypingProblem(Problem):
    """A problem that exposes a set of sbi files converted to .vec/.vecp format. Files must
    be named using the convention: basename-train.vec, basename-validation.vec basename-unlabeled.vec
    and have associated .vecp meta-data files.
    """

    def name(self):
        return "genotyping:" + self.basename

    def input_size(self, input_name):
        # TODO: implement for input_name = vector_name
        #vector_index = self.meta_data.get
        #self.meta_data.get_vector_dimensions_idx()
        return (361)

    def output_size(self, output_name):
        # TODO: implement for output_name = vector_name
        #vector_index = self.meta_data.get
        #self.meta_data.get_vector_dimensions_idx()
        return (5)

    def train_set(self):
        return GenotypeDataset(self.basename + "-train.vec", vector_names=["input", "softmaxGenotype"])

    def unlabeled_set(self):
        if self.file_exists(self.basename + "-unlabeled.vec"):
            return GenotypeDataset(self.basename + "-unlabeled.vec", vector_names=[ "input", "softmaxGenotype"])
        else:
            return EmptyDataset()

    def validation_set(self):
        if self.file_exists(self.basename + "-validation.vec"):
            return GenotypeDataset(self.basename + "-validation.vec",  vector_names=["input", "softmaxGenotype"])
        else:
            return EmptyDataset()

    def __init__(self, mini_batch_size, code, num_workers=0):
        super().__init__(mini_batch_size)
        self.basename = code[len("genotyping:"):]
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
