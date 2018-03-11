
import torch
import pickle
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import SmallerDataset


class Problem:
    def __init__(self, mini_batch_size=128):
        self._mini_batch_size = mini_batch_size

    def describe(self):
        print("{} Problem has {} training , {} validation and {} unlabeled examples".format(
            self.name(),
            len(self.train_loader()) * self.mini_batch_size(),
            len(self.validation_loader()) * self.mini_batch_size(),
            len(self.unlabeled_loader()) * self.mini_batch_size(),
        ))

    def load_tensor(self, vector_name, tensor_kind):
        """Return a tensor associated with this problem. Tensors are written to disk for specific problems, vectors and kinds of
        tensor (e.g., kind=mean stores the mean of each element of the tensor across a problem)."""
        with open(self.basename+"_"+vector_name+"."+tensor_kind, mode="rb") as tensor_file:
            return pickle.load(tensor_file)

    def name(self):
        pass

    def input_size(self, input_name):
        """Returns the shape of an input, e.g., (3,32,32) for a 3 channel image with dimensions
        32x32 pixels.
        """
        return (0, 0, 0)

    def output_size(self, output_name):
        """Returns the shape of an output, e.g., (10) for a one-hot encoded 10-class.
        """
        return (0, 0, 0)


    def mini_batch_size(self):
        return self._mini_batch_size

    def train_set(self):
        """Returns the training DataSet."""
        return None

    def unlabeled_set(self):
        """Returns the unsupervised DataSet."""
        return None

    def validation_set(self):
        """Returns the validation DataSet."""
        return None

    def test_set(self):
        """Returns the test DataSet."""
        return None

    def loader_for_dataset(self, dataset):
        pass

    def train_loader(self):
        """Returns the torch dataloader over the training set. """
        pass

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        pass

    def train_loader_subset_range(self, start, end):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        if start==0:
            return self.loader_for_dataset(SmallerDataset(delegate=self.train_set(), new_size=end),shuffle=True)
        else:
            return self.train_loader_subset(range(start, end))

    def train_loader_subset(self, indices):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the examples identified by these indices."""
        pass

    def validation_loader(self):
        """Returns the torch dataloader over the validation set. """
        pass

    def test_loader_subset(self, indices):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        pass

    def validation_loader_subset(self, indices):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        pass

    def validation_loader_range(self, start, end):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        if start==0:
            return self.loader_for_dataset(SmallerDataset(delegate=self.validation_set(), new_size=end))
        else:
            return self.validation_loader_subset(range(start,end))

    def test_loader_range(self, start, end):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        if start==0:
            return self.loader_for_dataset(SmallerDataset(delegate=self.test_set(), new_size=end))
        else:
            return self.test_loader_subset(range(start,end))


    def unlabeled_loader(self):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        pass

    def unlabeled_loader_subset(self, indices):
        """Returns the torch dataloader over the regularization set, shuffled,
        but limited to the example range start-end."""
        pass

    def unlabeled_loader_subset_range(self, start, end):
        """Returns the torch dataloader over the regularization set, shuffled,
        but limited to the example range start-end."""
        return self.unlabeled_loader_subset(range(start, end))

    def loss_function(self, output_name):
        """Return the loss function for this problem."""
        pass

    def model_attrs(self):
        """Return additional attributes that can be added to the model"""
        pass


