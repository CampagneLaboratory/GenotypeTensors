import unittest

from org.campagnelab.dl.genotypetensors.structured.BatchedSbiMappers import MapSequence, BatchedStructuredEmbedding


class MapContainer(BatchedStructuredEmbedding):

    def __init__(self, embedding_size, use_cuda=False):
        super().__init__(embedding_size, use_cuda)
        self.map_sequence=MapSequence()

    def collect_tensors(self, elements, field_prefix, tensors, index_map=[]):
        field_prefix+="container"
        for instance in elements:
            if 'indices' not in instance.keys():
                instance['indices']={}
        for instance in elements:
            self.map_sequence.collect_tensors(instance['sequence'], field_prefix, tensors,
                                          index_maps=instance['indices'])
        return tensors

    def forward_batch(self, elements, field_prefix, tensors,index_maps=[]):
        super().forward_batch(elements,field_prefix,tensors,index_maps)
        field_prefix += "container"
        # do forward on sequences:
        return self.map_sequence.forward_batch([instance['sequence'] for instance in elements ],field_prefix,tensors,
                                        index_maps=[instance['indices'] for instance in elements ])

class BatchedMappersTestCase(unittest.TestCase):
    def test_sequence(self):


        map_container =MapContainer(embedding_size=4)
        containers=[{'sequence':"ACTG"},
                   {'sequence':"A"} ]

        tensors=map_container.collect_tensors(containers, "",{})
        self.assertIsNotNone(tensors)
        self.assertEqual(containers[0]['indices']['container.sequence'],[0])
        self.assertEqual(containers[1]['indices']['container.sequence'],[1])

        self.assertEqual(containers[0]['indices']['container.sequence.bases'], [0,1,2,3])
        self.assertEqual(containers[1]['indices']['container.sequence.bases'], [4])

        # test forward batch:

        batch = map_container.forward_batch(containers, "", tensors)
        print(batch)
        self.assertIsNotNone(batch)
        self.assertEqual(batch.size(0),2)
if __name__ == '__main__':
    unittest.main()
