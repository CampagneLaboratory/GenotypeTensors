import unittest

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.BatchedSbiMappers import MapSequence, BatchedStructuredEmbedding, \
    MapCountInfo, MapSampleInfo, MapBaseInformation


class BatchedMappersTestCase(unittest.TestCase):
    def test_count(self):


        map_container =MapCountInfo()
        import ujson
        record = ujson.loads(sbi_json_string)
        containers=[record['samples'][0]['counts'][0],record['samples'][0]['counts'][1] ]

        tensors=map_container.collect_tensors(containers, "counts",tensors={})
        self.assertIsNotNone(tensors)
        self.assertEqual(containers[0]['indices']['counts.count.isIndel'],[0])
        self.assertEqual(containers[0]['indices']['counts.count.toSequence.sequence'],[0])

        self.assertEqual(containers[1]['indices']['counts.count.isIndel'], [1])
        self.assertEqual(containers[1]['indices']['counts.count.toSequence.sequence'], [1])

        # test forward batch:

        mapped = map_container.forward_batch(containers, "counts", tensors)
        print(mapped)
        self.assertIsNotNone(mapped)

    def test_sample(self):


        count_mapper =MapCountInfo()
        map_container =MapSampleInfo(count_mapper=count_mapper,count_dim=64, sample_dim=66, num_counts=2)
        import ujson
        record = ujson.loads(sbi_json_string)
        container=[record['samples'][0],record['samples'][0] ]

        tensors=map_container.collect_tensors(container, "samples",{})
        self.assertIsNotNone(tensors)
        # we must have collected two samples:
        self.assertEqual(container[0]['indices']['samples.sample.list'],[0,1])

        # test forward batch:

        mapped = map_container.forward_batch(container, "samples", tensors,index_maps=[sample['indices'] for sample in container])
        print(mapped)
        self.assertIsNotNone(mapped)

    def test_baseinfo(self):


        count_mapper =MapCountInfo()
        sample_mapper =MapSampleInfo(count_mapper=count_mapper,count_dim=64, sample_dim=66, num_counts=2)
        map_container =MapBaseInformation(sample_mapper=sample_mapper,sample_dim=sample_mapper.embedding_size,  num_samples=1, sequence_output_dim=64, ploidy=2, extra_genotypes=2)
        import ujson
        record = ujson.loads(sbi_json_string)
        container=[record,record ]

        tensors=map_container.collect_tensors(container, "records",{})
        self.assertIsNotNone(tensors)
        # we must have collected two samples:
        self.assertEqual(container[0]['indices']['records.record.list'],[0,1])

        # test forward batch:

        mapped = map_container.forward_batch(container, "records", tensors,index_maps=[record['indices'] for record in container])
        print(mapped)
        self.assertIsNotNone(mapped)

    def test_forward_baseinfo(self):


        count_mapper =MapCountInfo()
        sample_mapper =MapSampleInfo(count_mapper=count_mapper,count_dim=64, sample_dim=66, num_counts=2)
        map_container =MapBaseInformation(sample_mapper=sample_mapper,sample_dim=sample_mapper.embedding_size,  num_samples=1, sequence_output_dim=64, ploidy=2, extra_genotypes=2)
        import ujson
        record = ujson.loads(sbi_json_string)
        container=[record,record ]

        mapped=map_container.forward(record)
        self.assertIsNotNone(mapped)

if __name__ == '__main__':
    unittest.main()
