import sys
from pathlib import Path

import os
import torch
from torchnet.dataset.dataset import Dataset

from org.campagnelab.dl.genotypetensors.VectorCache import VectorCache
from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader
from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.VectorReaderBinary import VectorReaderBinary


class  SmallerDataset(Dataset):
    def __init__(self, delegate, new_size):
        assert new_size<=len(delegate) or new_size==sys.maxsize, "new size must be smaller or equal to delegate size."
        self.length=new_size
        self.delegate=delegate

    def __len__(self):
        return min(self.length,len(self.delegate))

    def __getitem__(self, idx):
        if idx <self.length:
            value=self.delegate[idx]
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
        self.reader_index=reader_index

        self.dataset=dataset
        self.num_elements=len(dataset)
        self.index_in_reader=0

    def at_end(self):
        return self.index_in_reader>=self.num_elements

    def advance(self):
        new_index=self.index_in_reader
        self.index_in_reader+=1
        return new_index

    def reset(self):
        self.index_in_reader=0

class InterleaveDatasets(Dataset):
    """ A dataset that exposes delegates by interleaving their records.
    """
    def __init__(self, dataset_list):
        self.dataset_list=dataset_list
        self.num_readers=len(dataset_list)
        self.index_delegates=[None]*self.num_readers
        for reader_index, dataset in enumerate(dataset_list):
            self.index_delegates[reader_index]=InterleavedReaderIndex(reader_index, dataset)

    def __len__(self):
        len=0
        for dataset in  self.dataset_list:
            len+=len(dataset)
        return len

    def __getitem__(self, idx):
        if len(self.index_delegates)==0:
            raise IndexError("CustomRange index out of range")
        reader_index= idx % len(self.index_delegates)
        delegate = self.index_delegates[reader_index]
        index_in_reader= delegate.advance()
        if delegate.at_end():
            self.index_delegates -= delegate
        return delegate[index_in_reader]


class CyclicInterleavedDatasets(Dataset):
    """ A dataset that exposes delegates by interleaving their records and cycling through them when no more
        items are available in a dataset.
    """
    def __init__(self, dataset_list):
        self.dataset_list=dataset_list
        self.num_readers=len(dataset_list)
        self.index_delegates=[None]*self.num_readers
        for reader_index, dataset in enumerate(dataset_list):
            self.index_delegates[reader_index]=InterleavedReaderIndex(reader_index, dataset)

    def __len__(self):
        """
        This Dataset returns sys.maxsize, but is effectively unlimited.
        :return:
        """
        return sys.maxsize

    def __getitem__(self, idx):

        reader_index= idx % len(self.index_delegates)
        delegate = self.index_delegates[reader_index]
        index_in_reader= delegate.advance()
        if delegate.at_end():
            delegate.reset()
        return delegate.dataset[index_in_reader]



class CachedGenotypeDataset(Dataset):
    def __init__(self, vec_basename,vector_names, max_records=sys.maxsize, sample_id=False):
        basename, file_extension = os.path.splitext(vec_basename)
        if not self.file_exists(basename+"-cached.vec") or not self.file_exists(basename+"-cached.vecp"):
            # Write cache:
            VectorCache(basename,max_records=max_records).write_lines()
        self.delegate=GenotypeDataset(basename+"-cached",vector_names, sample_id)

    def file_exists(self, filename):
        return Path(filename).is_file()

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx):
        return self.delegate[idx]

class GenotypeDataset(Dataset):
    """" Implement a dataset that can be traversed only once, in increasing and contiguous index number."""
    def __init__(self, vec_basename,  vector_names,sample_id=False):
        super().__init__()
        # initialize vector reader with basename and selected vectors:
        self.reader = VectorReader(vec_basename, sample_id=sample_id, vector_names=vector_names,
                                   return_example_id=True)
        self.props = self.reader.vector_reader_properties
        # obtain number of examples from .vecp file:
        self.length = self.props.num_records
        self.vector_names=vector_names
        self.is_random_access=self.props.file_type == "binary"
        self.previous_index=0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # get next example from .vec file and check that idx matches example index,
        # then return the features and outputs as tuple.
        if self.is_random_access:
            if idx!=(self.previous_index+1):
                self.reader._set_to_example_at_idx(idx)
            example_tuple = next(self.reader)
        else:
            example_tuple = next(self.reader)
            assert example_tuple[0] == idx, "Requested example index out of order of .vec file."

        result={}
        i=0
        for tensor in example_tuple[1:]:
            result[self.vector_names[i]]= torch.from_numpy(tensor)
            i+=1
        self.previous_index = idx
        return  result


class MultiProcessingPredictDataset(Dataset):
    """
    Implementation of dataset that supports multiple workers in predict phase by allowing each worker to separately
    create a pointer to the vector file when reading in data
    """
    def __init__(self, vec_basename, vector_names, sample_id=0, cache_first=True, max_records=sys.maxsize,
                 use_cuda=False):
        super().__init__()
        if cache_first:
            vector_path = "{}-test-cached.vec".format(vec_basename)
            properties_path = "{}-test-cached.vecp".format(vec_basename)
            if not os.path.isfile(vector_path) or not os.path.isfile(properties_path):
                with VectorCache(vec_basename, max_records) as vector_cache:
                    vector_cache.write_lines()
            vec_basename = "{}-test-cached".format(vec_basename)
        else:
            vec_basename = "{}-test".format(vec_basename)
            vector_path = "{}-test.vec".format(vec_basename)
            properties_path = "{}-test.vecp".format(vec_basename)
        vector_properties = VectorPropertiesReader(properties_path)
        num_bytes = VectorReaderBinary.check_file_size(vector_path, vector_properties.num_records,
                                                       vector_properties.num_bytes_per_example)
        self.reader = VectorReader(vec_basename, sample_id=sample_id, vector_names=vector_names,
                                   return_example_id=True, parallel=True, num_bytes=num_bytes)
        self.props = self.reader.vector_reader_properties
        if not self.props.file_type == "binary":
            raise TypeError("MultiProcessingPredictDataset expects a binary file but instead received a {} file"
                            .format(self.props.file_type))
        self.length = self.props.num_records
        self.vector_names = vector_names
        self.use_cuda = use_cuda

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        example_tuple = self.reader.__getitem__(idx)
        if example_tuple[0] != idx:
            raise Exception("Mismatch between requested example id {} and returned id {}".format(idx, example_tuple[0]))
        result = {}
        i = 0
        for tensor in example_tuple[1:]:
            var = torch.from_numpy(tensor)
            if self.use_cuda:
                var = var.cuda(async=True)
            result[self.vector_names[i]] = var
            i += 1
        return result
