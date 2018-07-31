import torch
import unittest

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo, MapSampleInfo

cuda=torch.device('cuda')
cpu=torch.device('cpu')
class PreloadTestCase(unittest.TestCase):
    def test_preload_count(self):
        map_container = MapCountInfo(device=cuda)
        import ujson
        record = ujson.loads(sbi_json_string)
        count = record['samples'][0]['counts'][0]
        loaded=map_container.preload(count)
        # move preloaded tensors to cuda:
        loaded.to(cuda)
        print(loaded)

        self.assertIsNotNone(loaded)
        mapped = loaded.mapper.loaded_forward(loaded)
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_preload_sample(self):
        count_mapper = MapCountInfo(device=cuda,count_dim=32)

        map_container = MapSampleInfo(count_mapper=count_mapper, count_dim=32, sample_dim=64, num_counts=3, device=cuda)
        import ujson
        record = ujson.loads(sbi_json_string)
        sample = record['samples'][0]
        loaded=map_container.preload(sample)
        # move preloaded tensors to cuda:
        loaded.to(cuda)
        print(loaded)

        self.assertIsNotNone(loaded)
        mapped = loaded.mapper.loaded_forward(loaded)
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)


if __name__ == '__main__':
    unittest.main()
