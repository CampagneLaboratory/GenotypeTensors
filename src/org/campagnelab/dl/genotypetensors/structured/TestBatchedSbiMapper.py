import unittest

import torch
from torch.nn import Module

from org.campagnelab.dl.genotypetensors.structured.Batcher import Batcher
from org.campagnelab.dl.genotypetensors.structured.Models import IntegerModel, NoCache, MeanOfList, BatchOfInstances, \
    StructuredEmbedding, map_Boolean
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo


class BatchedStructuredSbiMapperTestCase(unittest.TestCase):
    def test_integer_batching(self):
        mapper = IntegerModel(100, 2)
        batcher = Batcher()
        list_of_ints = [12, 3, 2]
        no_cache = NoCache()

        example_indices = []

        for value in list_of_ints:
            example_indices += [batcher.collect_inputs(mapper, [value])]

        print(batcher.get_batched_input(mapper))
        print(batcher.forward_batch(mapper))
        for example_index in example_indices:
            print(batcher.get_forward_for_example(mapper, example_index))

        self.assertEqual(str(mapper([12], no_cache).data),str(batcher.get_forward_for_example(mapper, 0).data))
        self.assertEqual(str(mapper([3], no_cache).data),str(batcher.get_forward_for_example(mapper, 1).data))
        self.assertEqual(str(mapper([2], no_cache).data),str(batcher.get_forward_for_example(mapper, 2).data))


    def test_boolean_batching(self):
        mapper = map_Boolean()
        batcher = Batcher()
        list_of_bools = [True,True,False]
        no_cache = NoCache()

        example_indices = []

        for value in list_of_bools:
            example_indices += [batcher.collect_inputs(mapper, value)]

        print("batched input={}".format(batcher.get_batched_input(mapper)))
        print(batcher.forward_batch(mapper))
        for example_index in example_indices:
            print("example {} = {}".format(example_index,batcher.get_forward_for_example(mapper, example_index)))

        self.assertEqual(str(mapper(True, no_cache).data),str(batcher.get_forward_for_example(mapper, 0).data))
        self.assertEqual(str(mapper(True, no_cache).data),str(batcher.get_forward_for_example(mapper, 1).data))
        self.assertEqual(str(mapper(False, no_cache).data),str(batcher.get_forward_for_example(mapper, 2).data))



    def test_message(self):
        torch.manual_seed(1212)

        class Message:
            def __init__(self, a, b):
                self.dict = {'type': "Message", 'a': a, 'b': b}

            def __getitem__(self, key):
                return self.dict[key]

        class MapMessage(StructuredEmbedding):
            def __init__(self):
                super().__init__(embedding_size=2)

                self.map_a = IntegerModel(distinct_numbers=100, embedding_size=2)
                self.map_b = IntegerModel(distinct_numbers=100, embedding_size=2)

            def forward(self, message,tensor_cache,cuda=None):
                return self.map_a([message['a']],tensor_cache,cuda) + self.map_b([message['b']],tensor_cache,cuda)

            def collect_inputs(self, message, phase=0, cuda=None, batcher=None):
                assert isinstance(message, Message)

                if phase == 0:
                    return {
                        id(self.map_a): self.map_a.collect_inputs(values=[message['a']], phase=phase, cuda=cuda),
                        id(self.map_b): self.map_b.collect_inputs(values=[message['b']], phase=phase, cuda=cuda)
                    }
                if phase == 1:
                    example_index = message['message_index']
                    my_a = batcher.get_forward_for_example(mapper=self.map_a, example_index=example_index)
                    my_b = batcher.get_forward_for_example(mapper=self.map_b, example_index=example_index)
                    return {
                        # combine individual examples, do sum the embeddings:
                        id(self): my_a + my_b
                    }

            def forward_batch(self, batcher, phase=0):
                my_input = {}
                if phase == 0:
                    # do forward for batches, results are kept in batcher.
                    batched_a = batcher.forward_batch(mapper=self.map_a)
                    batched_b = batcher.forward_batch(mapper=self.map_b)
                    return (batched_a, batched_b)

                if phase == 1:
                    return batcher.get_batched_input(mapper=self)

        messages = [Message(12, 4), Message(1, 6)]
        batcher = Batcher()
        mapper = MapMessage()
        no_cache=NoCache()
        for message in messages:
            message.dict['message_index'] = batcher.collect_inputs(mapper, message)

        batcher.forward_batch(mapper=mapper, phase=0)
        for message in messages:
            batcher.collect_inputs(mapper, message, phase=1)

        batcher.forward_batch(mapper=mapper, phase=1)
        for message in messages:
            print(batcher.get_forward_for_example(mapper, example_index=message.dict['message_index']))

        self.assertEqual(str(mapper(messages[0],no_cache).data),str(batcher.get_forward_for_example(mapper, example_index=0).data))
        self.assertEqual(str(mapper(messages[1],no_cache).data),str(batcher.get_forward_for_example(mapper, example_index=1).data))

    def test_count(self):
        json_string = '{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0}'
        import ujson
        count = ujson.loads(json_string)

        map_CountInfo = MapCountInfo(mapped_count_dim=5, count_dim=16, mapped_base_dim=6,
                                     mapped_genotype_index_dim=2)
        batcher=Batcher()
        count['count_index']=batcher.collect_inputs(map_CountInfo,count,phase=0)
        batcher.forward_batch(mapper=map_CountInfo)
        batcher.collect_inputs(map_CountInfo, count, phase=1)
        batcher.forward_batch(mapper=map_CountInfo,phase=1)
        print(batcher.get_forward_for_example(map_CountInfo,0))

    def test_two_count(self):
        json_string = '{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0}'
        import ujson
        count = ujson.loads(json_string)
        counts=[count]*2
        map_CountInfo = MapCountInfo(mapped_count_dim=5, count_dim=16, mapped_base_dim=6,
                                     mapped_genotype_index_dim=2)
        batcher = Batcher()
        for count in counts:
            count['count_index'] = batcher.collect_inputs(map_CountInfo, count, phase=0)
        batcher.forward_batch(mapper=map_CountInfo)
        for count in counts:
            batcher.collect_inputs(map_CountInfo, count, phase=1)
        batcher.forward_batch(mapper=map_CountInfo, phase=1)
        print(batcher.get_forward_for_example(map_CountInfo, 0))
        print(batcher.get_forward_for_example(map_CountInfo, 1))
if __name__ == '__main__':
    unittest.main()
