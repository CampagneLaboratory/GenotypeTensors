import unittest

import torch

from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapNumberWithFrequencyList


class NumberWithFrequencyTestCase(unittest.TestCase):
    def test_something(self):
        mapper=MapNumberWithFrequencyList(distinct_numbers=1000, mapped_number_dim=4)

        json_string='[{"type":"NumberWithFrequency","frequency":1,"number":23},{"type":"NumberWithFrequency","frequency":1,"number":30},{"type":"NumberWithFrequency","frequency":5,"number":34}]'
        import ujson
        nw_list = ujson.loads(json_string)

        result = mapper.forward(nwf_list=nw_list)
        print(result)
        self.assertEqual(torch.Size((1,7)), result.size())


if __name__ == '__main__':
    unittest.main()
