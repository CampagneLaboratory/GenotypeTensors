from torchnet.dataset.dataset import Dataset

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader

class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        pass

class GenotypeDataset(Dataset):
    """" Implement a dataset that can be traversed only once, in increasing and contiguous index number."""
    def __init__(self, vec_basename, sample_id, vector_names):
        super().__init__()
        # initialize vector reader with basename and selected vectors:
        self.reader = VectorReader(vec_basename, sample_id=sample_id, vector_names=vector_names,
                                   return_example_id=True)
        self.props = self.reader.vector_reader_properties
        # obtain number of examples from .vecp file:
        self.length = self.props.get_num_records()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get next example from .vec file and check that idx matches example index,
        # then return the features and outputs as tuple.
        example_tuple = next(self.reader)
        assert example_tuple[0] == idx, "Requested example index out of order of .vec file."
        return example_tuple[1:]
