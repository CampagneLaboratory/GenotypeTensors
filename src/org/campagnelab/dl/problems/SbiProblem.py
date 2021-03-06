import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import EmptyDataset, \
    ListDataset
from org.campagnelab.dl.problems.Problem import Problem


class SbiProblem(Problem):
    def get_vector_names(self):
        return []

    def get_input_names(self):
        return []

    def get_output_names(self):
        return []

    def input_size(self, input_name):
        return self.meta_data.get_vector_dimensions_from_name(input_name)

    def output_size(self, output_name):
        return self.meta_data.get_vector_dimensions_from_name(output_name)

    def train_set(self):
        return ListDataset(self.basename, "train", self.get_vector_names())

    def validation_set(self):
        return ListDataset(self.basename, "validation", self.get_vector_names())

    def test_set(self):
        return ListDataset(self.basename, "test", self.get_vector_names())

    def unlabeled_set(self):
        return ListDataset(self.basename, "unlabeled", self.get_input_names())

    def __init__(self, mini_batch_size, code, drop_last_batch=True, num_workers=0):
        super().__init__(mini_batch_size)
        self.basename = code[len(self.basename_prefix()):]
        self.num_workers = num_workers
        self.drop_last_batch = drop_last_batch
        self.reader = None
        self.meta_data = None
        for dataset in ["train", "validation", "test", "unlabeled"]:
            try:
                self.reader = VectorReader(self.basename + "-" + dataset, sample_id=0, return_example_id=False,
                                           vector_names=[])
                break
            except:
                pass
        self.load_metadata()

    def load_metadata(self):
        assert self.reader is not None, "Unable to load properties from any dataset (tried train, validation,test,unlabeled)"
        self.meta_data = self.reader.vector_reader_properties
        self.reader.close()

    def train_loader(self):
        """Returns the torch dataloader over the training set. """

        return self.loader_for_dataset(self.train_set(), shuffle=True)

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
        return self.loader_for_dataset(self.validation_set())

    def test_loader_subset(self, indices):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        assert False, "Not support for text .vec files"

    def unlabeled_loader(self):
        dataset = self.unlabeled_set()
        use_shuffle = len(dataset)>0
        return self.loader_for_dataset(dataset=dataset, shuffle=use_shuffle)

    def unlabeled_loader_subset(self, indices):
        """Returns the torch dataloader over the unlabeled set, limiting to the examples
        identified by the indices. """
        assert False, "Not support for text .vec files"

    def test_loader(self):
        return self.loader_for_dataset(dataset=self.test_set(), shuffle=False)

    def reg_loader_subset(self, indices):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        # transform the unsupervised set the same way as the training set:

        assert False, "Not support for text .vec files"

    def loader_for_dataset(self, dataset, shuffle=False):
        return iter(DataLoader(dataset=dataset, shuffle=shuffle, batch_size=self.mini_batch_size(),
                               num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last_batch))

    def loss_function(self, output_name):
        return torch.nn.CrossEntropyLoss()

    def file_exists(self, filename):
        return Path(filename).is_file()

    def basename_prefix(self):
        return None

    def model_attrs(self):
        return {
            "samples": self.meta_data.samples,
            "domain_descriptor": self.meta_data.vector_properties["domainDescriptor"],
            "input_files": copy.deepcopy(self.meta_data.vector_properties["inputFiles"]),
            "feature_mapper": self.meta_data.vector_properties["featureMapper"]
        }


class SbiSomaticProblem(SbiProblem):
    def name(self):
        return self.basename_prefix() + self.basename

    def get_vector_names(self):
        return ["input", "isBaseMutated", "somaticFrequency"]

    def get_input_names(self):
        return ["input"]

    def get_output_names(self):
        return ["isBaseMutated", "somaticFrequency"]

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

    def get_input_names(self):
        return ["input"]

    def get_vector_names(self):
        return ["input", "softmaxGenotype", "metaData"]

    def get_output_names(self):
        return ["softmaxGenotype"]
