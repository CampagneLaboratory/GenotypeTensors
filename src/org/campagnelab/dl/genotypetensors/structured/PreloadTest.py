import torch
import unittest

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo

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

        #self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
