import sys

import torch
from torchnet.dataset.dataset import Dataset

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader

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



class GenotypeDataset(Dataset):
    """" Implement a dataset that can be traversed only once, in increasing and contiguous index number."""
    def __init__(self, vec_basename,  vector_names,sample_id=False):
        super().__init__()
        # initialize vector reader with basename and selected vectors:
        self.reader = VectorReader(vec_basename, sample_id=sample_id, vector_names=vector_names,
                                   return_example_id=True)
        self.props = self.reader.vector_reader_properties
        # obtain number of examples from .vecp file:
        self.length = self.props.get_num_records()
        self.vector_names=vector_names

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get next example from .vec file and check that idx matches example index,
        # then return the features and outputs as tuple.
        if self.props.file_type == "text" or self.props.file_type == "gzipped+text":
            example_tuple = next(self.reader)
            assert example_tuple[0] == idx, "Requested example index out of order of .vec file."
        elif self.props.file_type == "binary":
            self.reader.set_to_example_at_idx(idx)
            example_tuple = next(self.reader)
        else:
            raise NotImplementedError
        result={}
        i=0
        for tensor in example_tuple[1:]:
            result[self.vector_names[i]]= torch.from_numpy(tensor)
            i+=1
        return  result
