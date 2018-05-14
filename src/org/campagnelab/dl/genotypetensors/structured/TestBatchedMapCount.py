import unittest

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.BatchedSbiMappers import MapSequence, BatchedStructuredEmbedding, \
    MapCountInfo


class BatchedMappersTestCase(unittest.TestCase):
    def test_count(self):


        map_container =MapCountInfo()
        import ujson
        record = ujson.loads(sbi_json_string)
        containers=[record['samples'][0]['counts'][0],record['samples'][0]['counts'][1] ]

        tensors=map_container.collect_tensors(containers, "",{})
        self.assertIsNotNone(tensors)
        self.assertEqual(containers[0]['indices']['count.isIndel'],[0])
        self.assertEqual(containers[0]['indices']['count.toSequence.sequence'],[0])

        self.assertEqual(containers[1]['indices']['count.isIndel'], [1])
        self.assertEqual(containers[1]['indices']['count.toSequence.sequence'], [1])

        # test forward batch:

        self.assertIsNotNone(map_container.forward_batch(containers,"",tensors))
if __name__ == '__main__':
    unittest.main()
