import unittest

from org.campagnelab.dl.genotypetensors.structured.BatchedSbiMappers import MapSequence, BatchedStructuredEmbedding, \
    MapNumberWithFrequencyList


class MapContainer(BatchedStructuredEmbedding):

    def __init__(self, embedding_size, use_cuda=False):
        super().__init__(embedding_size, use_cuda)
        self.map_nwl = MapNumberWithFrequencyList(distinct_numbers=100, mapped_number_dim=4)

    def collect_tensors(self, elements, field_prefix, tensors, index_maps={}):
        assert isinstance(index_maps,dict),"index_maps must be of type dict"
        field_prefix += "container."
        for instance in elements:
            if 'indices' not in instance.keys():
                instance['indices'] = {}
        for instance in elements:
            self.map_nwl.collect_tensors(instance['qualityScores'], field_prefix+'qualityScores', tensors,
                                          index_maps=instance['indices'] )
        return tensors

    def forward_batch(self, elements, field_prefix, tensors, index_maps={}):
        field_prefix += "container"
        # do forward on sequences:
        return self.map_nwl.forward_batch(elements=[container['qualityScores'] for container in elements],
                                          field_prefix=field_prefix+".qualityScores", tensors=tensors,
                                               index_maps=[instance['indices'] for instance in elements])


class BatchedMappersTestCase(unittest.TestCase):
    def test_sequence(self):
        map_container = MapContainer(embedding_size=4)
        containers = [{'qualityScores': [{'type':"NumberWithFrequency",'number': 1,  'frequency': 34},
                                         {'type':"NumberWithFrequency",'number': 10, 'frequency': 2}]},

                      {'qualityScores': [{'type':"NumberWithFrequency",'number': 2,  'frequency': 12},
                                         {'type':"NumberWithFrequency",'number': 12, 'frequency': 3}]},
                      {'qualityScores': []},
                      ]

        tensors = map_container.collect_tensors(containers, "", {})
        self.assertIsNotNone(tensors)
        self.assertEqual(containers[0]['indices']['container.qualityScores.nwf.frequency'], [0,1])
        self.assertEqual(containers[0]['indices']['container.qualityScores.nwf.number'], [0,1])

        self.assertEqual(containers[1]['indices']['container.qualityScores.nwf.frequency'], [2,3])
        self.assertEqual(containers[1]['indices']['container.qualityScores.nwf.number'], [2,3])

        # test forward batch:

        batch = map_container.forward_batch(containers, "", tensors, [container['indices'] for container in containers])
        print(batch)
        self.assertIsNotNone(batch)


if __name__ == '__main__':
    unittest.main()
