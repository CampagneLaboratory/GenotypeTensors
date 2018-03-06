import bisect
import sys
import threading
from pathlib import Path

import os

import numpy
import torch
from torchnet.dataset.dataset import Dataset

from org.campagnelab.dl.genotypetensors.VectorCache import VectorCache
from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader
from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.VectorReaderBinary import VectorReaderBinary

from multiprocessing import Lock, current_process


# Given n items and s sets to partition into, return ceiling of count in any partition (faster than math.ceil)
def _ceiling_partition(n, s):
    if s == 0:
        return n
    return -(-n // s)


class SmallerDataset(Dataset):
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
            return value
        else:
            assert False, "index is larger than trimmed length."


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass


class InterleavedReaderIndex:
    def __init__(self, reader_index, dataset):
        self.reader_index = reader_index
        self.dataset = dataset
        self.num_elements = len(dataset)
        self.index_in_reader = 0

    def at_end(self):
        return self.index_in_reader >= self.num_elements

    def advance(self):
        new_index = self.index_in_reader
        self.index_in_reader += 1
        return new_index

    def reset(self):
        self.index_in_reader = 0


class InterleaveDatasets(Dataset):
    """ A dataset that exposes delegates by interleaving their records.
    """
    def __init__(self, dataset_list):
        super().__init__()
        self.dataset_list = dataset_list
        self.num_readers = len(dataset_list)
        self.index_delegates = [InterleavedReaderIndex(reader_index, dataset)
                                for reader_index, dataset in enumerate(dataset_list)]

    def __len__(self):
        dataset_len = 0
        for dataset in self.dataset_list:
            dataset_len += len(dataset)
        return len

    def __getitem__(self, idx):
        if len(self.index_delegates) == 0:
            raise IndexError("CustomRange index out of range")
        reader_index = idx % len(self.index_delegates)
        delegate = self.index_delegates[reader_index]
        index_in_reader = delegate.advance()
        if delegate.at_end():
            self.index_delegates -= delegate
        return delegate[index_in_reader]


class CyclicInterleavedDatasets(Dataset):
    """ A dataset that exposes delegates by interleaving their records and cycling through them when no more
        items are available in a dataset.
    """
    def __init__(self, dataset_list):
        super().__init__()
        self.dataset_list = dataset_list
        self.num_readers = len(dataset_list)
        self.index_delegates = [InterleavedReaderIndex(reader_index, dataset)
                                for reader_index, dataset in enumerate(dataset_list)]

    def __len__(self):
        """
        This Dataset returns sys.maxsize, but is effectively unlimited.
        :return:
        """
        return sys.maxsize

    def __getitem__(self, idx):
        reader_index = idx % len(self.index_delegates)
        delegate = self.index_delegates[reader_index]
        index_in_reader = delegate.advance()
        if delegate.at_end():
            delegate.reset()
        return delegate.dataset[index_in_reader]


class ClippedDataset(Dataset):
    """
    A dataset that is clipped to bounds on the index.
    Used in combination with index dispatch to improve locality of access to a large dataset.
    """

    def __init__(self, delegate, num_slices, slice_index):
        super().__init__()
        assert 0 <= slice_index < num_slices, "slice_index must be between 0 and num_slices"
        self.start_index = 0
        self.delegate = delegate
        self.delegate_length = len(delegate)
        self.slice_length = int(self.delegate_length / num_slices)
        # if (slice_index == num_slices - 1):
        #     # last slice is smaller:
        #     self.slice_length = int(self.delegate_length - self.slice_length * (num_slices - 1))

        self.start_index = int(self.slice_length * slice_index)
        self.end_index = self.start_index + self.slice_length
        # print("index {} start: {} end: {} ", slice_index, self.start_index, self.end_index)

    def __len__(self):
        return self.slice_length

    def __getitem__(self, idx):
        if self.start_index <= idx < self.end_index:
            value = self.delegate[idx]
            return value
        else:
            assert False, "index is outside the clipped bounds: {} <= {} < {}.".format(self.start_index, idx,
                                                                                       self.end_index)


class CachedGenotypeDataset(Dataset):
    def __init__(self, vec_basename, vector_names, max_records=sys.maxsize, sample_id=0):
        super().__init__()
        basename, file_extension = os.path.splitext(vec_basename)
        if (not CachedGenotypeDataset.file_exists(basename + "-cached.vec")
                or not CachedGenotypeDataset.file_exists(basename + "-cached.vecp")):
            # Write cache:
            VectorCache(basename, max_records=max_records).write_lines()
        self.delegate = GenotypeDataset(basename + "-cached", vector_names, sample_id)
        self.basename = basename
        self.vector_names = vector_names
        self.sample_id = sample_id
        self.max_records = max_records

    @staticmethod
    def file_exists(filename):
        return Path(filename).is_file()

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        return self.delegate[idx]

    def slice(self, num_slices, slice_index):
        """Create a copy of this reader, focused on a slice of the data. """
        return ClippedDataset(CachedGenotypeDataset(self.basename, self.vector_names,
                                                    _ceiling_partition(len(self), num_slices),
                                                    self.sample_id),
                              num_slices=num_slices, slice_index=slice_index)


class GenotypeDataset(Dataset):
    """" Implement a dataset that can be traversed only once, in increasing and contiguous index number."""
    def __init__(self, vec_basename, vector_names, sample_id=0):
        super().__init__()
        self.props = VectorPropertiesReader("{}.vecp".format(os.path.splitext(vec_basename)[0]))
        # obtain number of examples from .vecp file:
        self.length = self.props.num_records
        self.vector_names = vector_names
        self.is_random_access = self.props.file_type == "binary"
        self.previous_index = 0
        # Delegate reader lazily created in first __getitem__ call- for multiprocessing
        self.reader = None
        self.vec_basename = vec_basename
        self.sample_id = sample_id
        self.vector_names = vector_names

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert 0 <= idx < self.length, "index {} out of reader bounds {} {}.".format(idx, 0, self.length)
        # get next example from .vec file and check that idx matches example index,
        # then return the features and outputs as tuple.
        # Lazily create delegate reader- for multiprocessing
        if self.reader is None:
            self.reader = VectorReader(self.vec_basename, sample_id=self.sample_id, vector_names=self.vector_names,
                                       return_example_id=True)
        if self.is_random_access:
            if idx != (self.previous_index + 1):
                self.reader.set_to_example_at_idx(idx)
            example_tuple = next(self.reader)
        else:
            example_tuple = next(self.reader)
            assert example_tuple[0] == idx, "Requested example index out of order of .vec file."
        result = {}
        i = 0
        for tensor in example_tuple[1:]:
            result[self.vector_names[i]] = torch.from_numpy(tensor)
            i += 1
        self.previous_index = idx
        return result


class DispatchDataset(Dataset):
    def __init__(self, base_delegate, num_workers):
        super().__init__()
        num_workers = max(1, num_workers)
        self.base_delegate = base_delegate
        self.delegate_readers = [self.base_delegate.slice(slice_index=worker_idx, num_slices=num_workers) for worker_idx
                                 in range(num_workers)]
        self.delegate_locks = [Lock() for _ in range(num_workers)]
        self.delegate_cumulative_sizes = numpy.cumsum([len(x) for x in self.delegate_readers])

    def __len__(self):
        return len(self.base_delegate)

    def __getitem__(self, idx):
        worker_index = self.delegate_cumulative_sizes.searchsorted(idx, 'right')
        # print("Process {} is requesting idx {}".format(current_process().pid, idx))
        with self.delegate_locks[worker_index]:
            result = self.delegate_readers[worker_index][idx]
        # print("RETURNED process {} idx {} worker_idx {}".format(current_process().pid, idx, worker_index))
        return result
