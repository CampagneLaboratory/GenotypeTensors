import os
import sys

from torchnet.dataset.dataset import Dataset

from org.campagnelab.dl.genotypetensors.SBIToJsonIterator import sbi_json_generator, SbiToJsonGenerator
from org.campagnelab.dl.genotypetensors.VectorCache import VectorCache
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import GenotypeDataset, CachedGenotypeDataset


class StructuredGenotypeDataset(Dataset):
    """Dataset to load an sbi record (JSON object)"""
    def __init__(self, sbi_basename, vector_names=None, max_records=sys.maxsize, sample_id=0):
        super().__init__()
        basename, file_extension = os.path.splitext(sbi_basename)
        # load only the labels and metaData from the vec file if vector_names is not None (not present for unlabeled):
        if vector_names is not None:
            try:
                if (not CachedGenotypeDataset.file_exists(basename + "-cached.vec")
                        or not CachedGenotypeDataset.file_exists(basename + "-cached.vecp")):
                    # Write cache:
                    with VectorCache(basename, max_records=max_records) as vector_cache:
                        vector_cache.write_lines()
                self.delegate_labels = GenotypeDataset(basename+"-cached.vec", vector_names=vector_names,sample_id= sample_id)
            except FileNotFoundError as e:
                raise Exception("Unable to find vec/vecp files with basename "+str(basename))
        self.delegate_features=JsonGenotypeDataset(min(max_records,len(self.delegate_labels)),basename=basename)
        self.basename = basename
        self.vector_names = vector_names
        self.sample_id = sample_id
        self.max_records = max_records

    def __len__(self):
        return len(self.delegate_labels)

    def __getitem__(self, idx):
        return (self.delegate_features[idx], self.delegate_labels[idx])

    def close(self):
        #self.delegate_labels.close()
        self.delegate_features.close()

class JsonGenotypeDataset(Dataset):
    def __init__(self, length, basename):
        super().__init__()
        self.length=length
        self.basename=basename
        self.generator=None
        self.generator_iter=None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx==0 or self.generator is None:
            if self.generator is not None:
                self.generator.close()
            self.generator = SbiToJsonGenerator(sbi_path=self.basename + ".sbi", sort=True,num_records=self.length,
                                                include_frequencies=True)
            self.generator_iter=iter(self.generator)
            idx=0
        if idx>=self.length:
            self.generator.close()
            raise StopIteration
        else:
            return next(self.generator_iter)

    def __del__(self):
        """Destructor for cases when the dataset is used inside an iterator. """
        if self.generator is not None:
            self.generator.close()
            del self.generator

    def close(self):
        """Destructor for cases when the dataset is used inside an iterator. """
        if self.generator is not None:
            self.generator.close()
