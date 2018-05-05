import os
import sys

from torchnet.dataset.dataset import Dataset

from org.campagnelab.dl.genotypetensors.SBIToJsonIterator import sbi_json_generator
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import GenotypeDataset


class StructuredGenotypeDataset(Dataset):
    """Dataset to load an sbi record (JSON object)"""
    def __init__(self, sbi_basename, vector_names=None, max_records=sys.maxsize, sample_id=0):
        super().__init__()
        basename, file_extension = os.path.splitext(sbi_basename)
        # load only the labels and metaData from the vec file:
        if vector_names is None:
            vector_names=["softmaxGenotype","metaData"]
        try:
            self.delegate_labels = GenotypeDataset(basename+".vec", vector_names=vector_names,sample_id= sample_id)
        except FileNotFoundError as e:
            raise Exception("Unable to find vec/vecp files with basename "+str(basename))
        self.delegate_features=JsonGenotypeDataset(len(self.delegate_labels),basename=basename)
        self.basename = basename
        self.vector_names = vector_names
        self.sample_id = sample_id
        self.max_records = max_records

    def __len__(self):
        return len(self.delegate_labels)

    def __getitem__(self, idx):
        return (self.delegate_features[idx], self.delegate_labels[idx])


class JsonGenotypeDataset(Dataset):
    def __init__(self, length, basename):
        super().__init__()
        self.length=length
        self.basename=basename

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx==0:
            self.generator=sbi_json_generator(sbi_path=self.basename+".sbi",sort=True)
        if idx>=self.length:
            del self.generator
            raise StopIteration
        else:
            return next(self.generator)

    def __del__(self):
        """Destructor for cases when the dataset is used inside an iterator. """
        if self.generator is not None:
            del self.generator