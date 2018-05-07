import unittest

from org.campagnelab.dl.genotypetensors.SBIToJsonIterator import SbiToJsonGenerator


class SbiToJSONTestCase(unittest.TestCase):
    @staticmethod
    def test_iter_without_cache():
        generator = SbiToJsonGenerator(sbi_path="/data/LSTM/NA12878_S1_gatk_realigned_filtered-2017-09-14-test.sbi",
                                       sort=True)
        for index, element in enumerate(generator):
            print(element)


if __name__ == '__main__':
    unittest.main()
