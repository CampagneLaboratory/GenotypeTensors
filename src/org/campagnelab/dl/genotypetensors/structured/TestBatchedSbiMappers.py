import unittest

from org.campagnelab.dl.genotypetensors.structured.BatchedSbiMappers import MapSequence, BatchedStructuredEmbedding


class MapContainer(BatchedStructuredEmbedding):

    def __init__(self, embedding_size, use_cuda=False):
        super().__init__(embedding_size, use_cuda)
        self.map_sequence=MapSequence()

    def collect_tensors(self, list, field_prefix, tensors, index_map=[]):
        field_prefix+="container"
        for instance in list:
            if 'indices' not in instance.keys():
                instance['indices']={}

        self.map_sequence.collect_tensors( [instance['sequence'] for instance in list ], field_prefix, tensors,
                           index_maps=[instance['indices'] for instance in list])
        return tensors

class BatchedMappersTestCase(unittest.TestCase):
    def test_sequence(self):


        map_container =MapContainer(embedding_size=4)
        containers=[{'sequence':"ACTG"},
                   {'sequence':"A"} ]

        tensors=map_container.collect_tensors(containers, "",{})
        self.assertIsNotNone(tensors)
        self.assertEqual(containers[0]['indices']['container.sequence'],[0])
        self.assertEqual(containers[1]['indices']['container.sequence'],[1])

        self.assertEqual(containers[0]['indices']['container.sequence.ints'], [0,1,2,3])
        self.assertEqual(containers[1]['indices']['container.sequence.ints'], [4])

if __name__ == '__main__':
    unittest.main()
