import unittest

import os

from org.campagnelab.dl.genotypetensors.SBIToJsonIterator import SbiToJsonGenerator


class SbiToJSONTestCase(unittest.TestCase):
    @staticmethod
    def test_iter_without_cache():
        sbi_path = "/data/LSTM/NA12878_S1_gatk_realigned_filtered-2017-09-14-test.sbi"
        if not os.path.exists(sbi_path):
            sbi_path = "data/LSTM/NA12878_S1_gatk_realigned_filtered-2017-09-14-test.sbi"
        generator = SbiToJsonGenerator(sbi_path=sbi_path, sort=True, num_records=324)
        for index, element in enumerate(generator):
            print(element)
        generator.close()

    @staticmethod
    def test_iter_with_cache():
        sbi_path = "/data/LSTM/NA12878_S1_gatk_realigned_filtered-2017-09-14-test.sbi"
        if not os.path.exists(sbi_path):
            sbi_path = "data/LSTM/NA12878_S1_gatk_realigned_filtered-2017-09-14-test.sbi"
        generator = SbiToJsonGenerator(sbi_path=sbi_path, sort=True, num_records=324, use_cache=True)
        for index, element in enumerate(generator):
            print(element)
        generator.close()


if __name__ == '__main__':
    unittest.main()
