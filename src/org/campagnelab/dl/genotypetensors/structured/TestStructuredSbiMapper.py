import pickle
import unittest

import torch

from org.campagnelab.dl.genotypetensors.structured.Models import BatchOfInstances
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import configure_mappers

def to_profile():
    json_string = '{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4}]}]}'
    import ujson
    record = ujson.loads(json_string)
    print(record)
    mappers, all_modules = configure_mappers(ploidy=2, extra_genotypes=3, num_samples=1)

    mapper = BatchOfInstances(mappers=mappers, all_modules=all_modules)
    mapper = mapper
    tensor_cache = TensorCache()
    mapper([record]*1000, tensor_cache=tensor_cache, cuda=False)

class StructuredSbiMapperTestCase(unittest.TestCase):
    def test_mapper(self):
        json_string='{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4}]}]}'
        import ujson
        record = ujson.loads(json_string)
        print(record)
        mappers, all_modules=configure_mappers(ploidy=2, extra_genotypes=3, num_samples=1)
        mapper = BatchOfInstances(mappers=mappers,all_modules=all_modules)
        out = mapper([record],tensor_cache=NoCache())
        print(out)

    def test_mapper_number_freqs(self):
        json_string='{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0,"qualityScoresForwardStrand":[{"type":"NumberWithFrequency","frequency":7,"number":40}],"qualityScoresReverseStrand":[{"type":"NumberWithFrequency","frequency":32,"number":40}],"readIndicesForwardStrand":[{"type":"NumberWithFrequency","frequency":1,"number":23},{"type":"NumberWithFrequency","frequency":1,"number":30},{"type":"NumberWithFrequency","frequency":5,"number":34}],"readIndicesReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":6},{"type":"NumberWithFrequency","frequency":1,"number":22},{"type":"NumberWithFrequency","frequency":1,"number":28},{"type":"NumberWithFrequency","frequency":1,"number":31},{"type":"NumberWithFrequency","frequency":1,"number":34},{"type":"NumberWithFrequency","frequency":1,"number":35},{"type":"NumberWithFrequency","frequency":1,"number":44},{"type":"NumberWithFrequency","frequency":1,"number":145},{"type":"NumberWithFrequency","frequency":1,"number":150},{"type":"NumberWithFrequency","frequency":5,"number":151},{"type":"NumberWithFrequency","frequency":2,"number":171},{"type":"NumberWithFrequency","frequency":4,"number":172}],"queryPositions":[{"type":"NumberWithFrequency","frequency":39,"number":0}],"pairFlags":[{"type":"NumberWithFrequency","frequency":6,"number":16},{"type":"NumberWithFrequency","frequency":14,"number":83},{"type":"NumberWithFrequency","frequency":6,"number":99},{"type":"NumberWithFrequency","frequency":12,"number":147},{"type":"NumberWithFrequency","frequency":1,"number":163}],"distancesToReadVariationsForwardStrand":[{"type":"NumberWithFrequency","frequency":2,"number":-70},{"type":"NumberWithFrequency","frequency":4,"number":-29}],"distancesToReadVariationsReverseStrand":[{"type":"NumberWithFrequency","frequency":2,"number":-24},{"type":"NumberWithFrequency","frequency":1,"number":-15},{"type":"NumberWithFrequency","frequency":1,"number":-2},{"type":"NumberWithFrequency","frequency":1,"number":12},{"type":"NumberWithFrequency","frequency":1,"number":13},{"type":"NumberWithFrequency","frequency":1,"number":15},{"type":"NumberWithFrequency","frequency":13,"number":29},{"type":"NumberWithFrequency","frequency":1,"number":49},{"type":"NumberWithFrequency","frequency":3,"number":62},{"type":"NumberWithFrequency","frequency":9,"number":70},{"type":"NumberWithFrequency","frequency":1,"number":73}],"distanceToStartOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":18},{"type":"NumberWithFrequency","frequency":1,"number":23},{"type":"NumberWithFrequency","frequency":1,"number":26},{"type":"NumberWithFrequency","frequency":1,"number":30},{"type":"NumberWithFrequency","frequency":30,"number":33},{"type":"NumberWithFrequency","frequency":5,"number":34}],"distanceToEndOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":6},{"type":"NumberWithFrequency","frequency":1,"number":22},{"type":"NumberWithFrequency","frequency":1,"number":28},{"type":"NumberWithFrequency","frequency":1,"number":31},{"type":"NumberWithFrequency","frequency":1,"number":34},{"type":"NumberWithFrequency","frequency":2,"number":35},{"type":"NumberWithFrequency","frequency":1,"number":44},{"type":"NumberWithFrequency","frequency":1,"number":48},{"type":"NumberWithFrequency","frequency":1,"number":49},{"type":"NumberWithFrequency","frequency":1,"number":50},{"type":"NumberWithFrequency","frequency":1,"number":52},{"type":"NumberWithFrequency","frequency":1,"number":62},{"type":"NumberWithFrequency","frequency":1,"number":63},{"type":"NumberWithFrequency","frequency":1,"number":68},{"type":"NumberWithFrequency","frequency":2,"number":75},{"type":"NumberWithFrequency","frequency":2,"number":76},{"type":"NumberWithFrequency","frequency":1,"number":81},{"type":"NumberWithFrequency","frequency":1,"number":83},{"type":"NumberWithFrequency","frequency":1,"number":88},{"type":"NumberWithFrequency","frequency":1,"number":89},{"type":"NumberWithFrequency","frequency":1,"number":100},{"type":"NumberWithFrequency","frequency":1,"number":104},{"type":"NumberWithFrequency","frequency":1,"number":109},{"type":"NumberWithFrequency","frequency":1,"number":111},{"type":"NumberWithFrequency","frequency":1,"number":117},{"type":"NumberWithFrequency","frequency":1,"number":118},{"type":"NumberWithFrequency","frequency":1,"number":121},{"type":"NumberWithFrequency","frequency":1,"number":125},{"type":"NumberWithFrequency","frequency":1,"number":128},{"type":"NumberWithFrequency","frequency":1,"number":133},{"type":"NumberWithFrequency","frequency":2,"number":138},{"type":"NumberWithFrequency","frequency":4,"number":139}]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":7}],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":115}],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":60}],"numVariationsInReads":[{"type":"NumberWithFrequency","frequency":1,"number":2}],"insertSizes":[{"type":"NumberWithFrequency","frequency":1,"number":-301}],"targetAlignedLengths":[{"type":"NumberWithFrequency","frequency":2,"number":148}],"queryAlignedLengths":[{"type":"NumberWithFrequency","frequency":1,"number":148}],"queryPositions":[{"type":"NumberWithFrequency","frequency":1,"number":0}],"pairFlags":[{"type":"NumberWithFrequency","frequency":1,"number":147}],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":-29},{"type":"NumberWithFrequency","frequency":1,"number":0}],"distanceToStartOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":33}],"distanceToEndOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":115}]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]}]}]}' \
                    ''
        import ujson
        record = ujson.loads(json_string)
        print(record)
        mappers, all_modules=configure_mappers(ploidy=2, extra_genotypes=3, num_samples=1)
        mapper = BatchOfInstances(mappers=mappers,all_modules=all_modules)
        out = mapper([record],tensor_cache=NoCache())
        print(out)

    def test_serialize_mapper(self):
        json_string='{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4}]}]}'
        import ujson
        record = ujson.loads(json_string)
        print(record)
        mappers, all_modules=configure_mappers(ploidy=2, extra_genotypes=3, num_samples=1)
        mapper = BatchOfInstances(mappers=mappers,all_modules=all_modules)
        with open('testing.pickle', 'wb') as f:
            pickle.dump(mapper,f)

    def test_train_mapper_on_cuda(self):
        json_string='{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4}]}]}'
        import ujson
        record = ujson.loads(json_string)
        print(record)
        mappers, all_modules=configure_mappers(ploidy=2, extra_genotypes=3, num_samples=1)

        mapper = BatchOfInstances(mappers=mappers,all_modules=all_modules)
        mapper=mapper.to(torch.device("cuda"))
        mapper([record],tensor_cache=NoCache(), cuda=True)

    def test_profile(self):
        import cProfile
        import unittest
        cProfile.run('from TestStructuredSbiMapper import to_profile; to_profile()')

    def test_train_mapper_on_cuda_with_cache(self):
        json_string = '{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4}]}]}'
        import ujson
        record = ujson.loads(json_string)
        print(record)
        mappers, all_modules = configure_mappers(ploidy=2, extra_genotypes=3, num_samples=1,device=torch.device('cuda'))

        mapper = BatchOfInstances(mappers=mappers, all_modules=all_modules,device=torch.device('cuda'))

        mapper([record])
if __name__ == '__main__':
    unittest.main()
