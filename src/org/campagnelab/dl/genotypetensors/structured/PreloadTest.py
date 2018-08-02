import torch
import unittest

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo, MapSampleInfo, MapBaseInformation

cuda=torch.device('cuda')
cpu=torch.device('cpu')
device=cpu
class PreloadTestCase(unittest.TestCase):
    def test_preload_count(self):
        map_container = MapCountInfo(device=device)
        import ujson
        record = ujson.loads(sbi_json_string)
        count = record['samples'][0]['counts'][0]
        loaded=map_container.preload(count)
        # move preloaded tensors to device:
        loaded.to(device)
        print(loaded)

        self.assertIsNotNone(loaded)
        mapped = loaded.mapper.loaded_forward(loaded)
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_preload_sample(self):
        count_mapper = MapCountInfo(device=device,count_dim=32)

        map_container = MapSampleInfo(count_mapper=count_mapper, count_dim=32, sample_dim=64, num_counts=3, device=device)
        import ujson
        record = ujson.loads(sbi_json_string)
        sample = record['samples'][0]
        loaded=map_container.preload(sample)
        # move preloaded tensors to device:
        loaded.to(device)
        print(loaded)

        self.assertIsNotNone(loaded)
        mapped = loaded.mapper.loaded_forward(loaded)
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_preload_record(self):
        count_mapper = MapCountInfo(device=device,count_dim=32)
        map_sample = MapSampleInfo(count_mapper=count_mapper, count_dim=32, sample_dim=64, num_counts=3, device=device)
        map_container=MapBaseInformation(sample_mapper=map_sample, sample_dim=64, num_samples=1, device=device)
        import ujson
        record = ujson.loads(sbi_json_string)

        loaded=map_container.preload(record)
        # move preloaded tensors to device:
        loaded.to(device)
        print(loaded)

        self.assertIsNotNone(loaded)
        mapped = loaded.mapper.loaded_forward(loaded)
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)


if __name__ == '__main__':
    unittest.main()
