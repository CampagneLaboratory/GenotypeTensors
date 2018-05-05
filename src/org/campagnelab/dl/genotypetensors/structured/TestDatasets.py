import unittest

from tqdm import tqdm

from org.campagnelab.dl.genotypetensors.structured.Datasets import StructuredGenotypeDataset
from org.campagnelab.dl.problems.StructuredSbiProblem import StructuredSbiGenotypingProblem


class StructuredDatasetTestCase(unittest.TestCase):
    def test_structured(self):
        dataset=StructuredGenotypeDataset(sbi_basename="/data/LSTM/NA12878_S1_gatk_realigned_filtered-2017-09-14-test",
                                          vector_names=["softmaxGenotype", "metaData"])
        example_0 = dataset[0]
        self.assertIsNotNone(example_0)
        sbi, (label, metaData) = dataset[1]
        self.assertIsNotNone(sbi)

        print("sbi:" + str(sbi))
        print("label:" + str(label))
        print("metaData:"+str(metaData))
        # uncomment to benchmark performance:
        #for index in tqdm(range(2,len(dataset))):
        #    dataset[index]

    def test_dataloader(self):
            problem=StructuredSbiGenotypingProblem(code="struct_genotyping:/data/LSTM/NA12878_S1_gatk_realigned_filtered-2017-09-14",
                                                   mini_batch_size=3)
            loader=problem.loader_for_dataset(problem.test_set())
            iterator=iter(loader)
            for _ in range(3):
                print(next(iterator))

if __name__ == '__main__':
    unittest.main()
