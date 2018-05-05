import unittest

import torch

from org.campagnelab.dl.genotypetensors.structured.Models import IntegerModel, MeanOfList, RNNOfList, BatchOfInstances, \
    Reduce


class IntegerEmbeddingTestCase(unittest.TestCase):
    def test_int_single(self):
        visitor = IntegerModel(100, 2)
        print(visitor([12, 3, 2]))

    def test_int_average(self):
        visitor_int = IntegerModel(100, 2)
        visitor_list_average = MeanOfList()
        print(visitor_list_average(visitor_int([12, 3, 2])))

    def test_int_rnn(self):
        torch.manual_seed(1212)
        visitor_int = IntegerModel(100, 2)
        visitor_list_rnn = RNNOfList(embedding_size=2, hidden_size=4, num_layers=1)
        print(visitor_list_rnn(visitor_int([12, 3, 2])))

    def test_message(self):
        torch.manual_seed(1212)

        class Message:
            def __init__(self, a, list_of_elements):
                self.dict = {'type': "Message", 'a': a, "list": list_of_elements}

            def __getitem__(self, key):
                return self.dict[key]

        def map_Message(m):
            mapped_a = map_a([m['a']])
            mapped_b = map_b(map_a(m['list']))

            return torch.cat([mapped_a, mapped_b], dim=0).view(1, -1)

        mappers = {"Message": map_Message}
        map_a = IntegerModel(distinct_numbers=100, embedding_size=2)
        map_b = MeanOfList()

        messages = [Message(12, [1, 5, 4]), Message(1, [2, 4, 1])]
        mapper = BatchOfInstances(mappers=mappers,all_modules=[map_a,map_b])
        out = mapper(messages)
        self.assertEquals(out.size(), (2, 4))

    def test_message_with_reduce(self):
            torch.manual_seed(1212)

            class Message:
                def __init__(self, a, list_of_elements):

                    self.dict={'type':"Message", 'a':a, "list":list_of_elements}

                def __getitem__(self, key):

                    return self.dict[key]

            reduce = Reduce([2, 2], 3)

            def map_Message(m):
                mapped_a = map_a([m['a']])
                mapped_b = map_b(map_a(m['list']))

                return reduce([mapped_a, mapped_b])

            mappers = {"Message": map_Message}
            map_a = IntegerModel(distinct_numbers=100, embedding_size=2)
            map_b = MeanOfList()

            messages = [Message(12, [1, 5, 4]), Message(1, [2, 4, 1])]
            mapper = BatchOfInstances(mappers=mappers,all_modules=[map_a,map_b])
            out = mapper(messages)
            self.assertEquals(out.size(), (2, 3))
            print(out)



if __name__ == '__main__':
    unittest.main()
