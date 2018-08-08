import torch
import unittest

from torchfold import torchfold

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string, \
    FoldExecutor
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo, MapSampleInfo, MapBaseInformation, \
    MapNumberWithFrequencyList, BatchOfRecords

cuda = torch.device('cuda')
cpu = torch.device('cpu')

device = cpu
class PreloadTestCase(unittest.TestCase):


    def test_fold_nwl(self):

        nwl_mapper=MapNumberWithFrequencyList(distinct_numbers=1000, device=device)
        import ujson
        count_mapper = MapCountInfo(device=device)

        fold = torchfold.Fold()
        mapped_nodes = []

        record = ujson.loads(sbi_json_string)
        count = record['samples'][0]['counts'][0]
        nwl=count['distanceToStartOfRead']
        loaded = nwl_mapper.preload(nwl,10)
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root_count_distanceToStartOfRead", loaded))
        mapped_nodes.append(loaded.mapper.fold(fold, "root_count_distanceToStartOfRead", loaded))
        mapped_nodes.append(loaded.mapper.fold(fold, "root_count_distanceToStartOfRead", loaded))
        excutor = FoldExecutor(count_mapper=count_mapper,sample_mapper=None, record_mapper=None)
        print(fold)
        mapped = fold.apply(excutor, [mapped_nodes])
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_fold_nwl2(self):

        nwl_mapper=MapNumberWithFrequencyList(distinct_numbers=1000, device=device)
        import ujson
        count_mapper = MapCountInfo(device=device)

        fold = torchfold.Fold()
        mapped_nodes = []

        record = ujson.loads(sbi_json_string)
        count = record['samples'][0]['counts'][0]
        nwl=count['distanceToStartOfRead']
        loaded = nwl_mapper.preload(nwl,10)
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root_count_numVariationsInReads", loaded))
        mapped_nodes.append(loaded.mapper.fold(fold, "root_count_numVariationsInReads", loaded))
        mapped_nodes.append(loaded.mapper.fold(fold, "root_count_numVariationsInReads", loaded))
        excutor = FoldExecutor(count_mapper=count_mapper,sample_mapper=None, record_mapper=None)
        print(fold)
        mapped = fold.apply(excutor, [mapped_nodes])
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_fold_count(self):

        count_mapper = MapCountInfo(device=device)
        import ujson


        fold = torchfold.Fold()
        mapped_nodes = []

        record = ujson.loads(sbi_json_string)
        count = record['samples'][0]['counts'][0]
        loaded = count_mapper.preload(count)
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root", loaded))
        excutor = FoldExecutor(count_mapper,sample_mapper=None, record_mapper=None)
        print(fold)
        mapped = fold.apply(excutor, [mapped_nodes])
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_fold_sample(self):
        count_mapper = MapCountInfo(device=device,count_dim=32)

        sample_mapper = MapSampleInfo(count_mapper=count_mapper, count_dim=32, sample_dim=64, num_counts=3, device=device)
        import ujson
        executor = FoldExecutor(count_mapper=count_mapper, sample_mapper=sample_mapper, record_mapper=None)


        fold = torchfold.Fold(executor)
        mapped_nodes = []

        record = ujson.loads(sbi_json_string)
        sample = record['samples'][0]
        loaded = sample_mapper.preload(sample)
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root", loaded))
        print(fold)
        mapped = fold.apply(executor, [mapped_nodes])
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)


    def test_fold_record(self):
        count_mapper = MapCountInfo(device=device,count_dim=32)

        sample_mapper = MapSampleInfo(count_mapper=count_mapper, count_dim=32, sample_dim=64, num_counts=3, device=device)
        record_mapper = MapBaseInformation(sample_mapper=sample_mapper, sample_dim=64, num_samples=1, ploidy=2,
                                           extra_genotypes=2, device = device)

        import ujson
        executor = FoldExecutor(count_mapper=count_mapper, sample_mapper=sample_mapper,
                                record_mapper=record_mapper)


        fold = torchfold.Fold(executor)
        mapped_nodes = []

        record = ujson.loads(sbi_json_string)

        loaded = record_mapper.preload(record)
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root", loaded))
        print(fold)
        mapped = fold.apply(executor, [mapped_nodes])
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_fold_batch_of_records(self):

        count_mapper = MapCountInfo(device=device,count_dim=32)
        sample_mapper = MapSampleInfo(count_mapper=count_mapper, count_dim=32, sample_dim=64, num_counts=3, device=device)
        record_mapper = MapBaseInformation(sample_mapper=sample_mapper, sample_dim=64, num_samples=1, ploidy=2,
                                           extra_genotypes=2, device = device)
        batch_of_records = BatchOfRecords(record_mapper,device = device)
        import ujson
        executor = FoldExecutor(count_mapper=count_mapper, sample_mapper=sample_mapper,
                                record_mapper=record_mapper)


        fold = torchfold.Fold(executor)
        mapped_nodes = []

        record = ujson.loads(sbi_json_string)

        loaded = batch_of_records.preload([record,record])
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root", loaded))
        print(fold)
        mapped = fold.apply(executor, mapped_nodes)
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

    def test_custom_sbi(self):
        test_json='{"type":"BaseInformation","referenceBase":"C","genomicSequenceContext":"AGGTCTCCTGGTCACCTAATCTTTTTTTTTTTTATACTTTA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"C","toSequence":"C","genotypeCountForwardStrand":2,"genotypeCountReverseStrand":16,"gobyGenotypeIndex":2,"qualityScoresForwardStrand":[{"type":"NumberWithFrequency","frequency":2,"number":40}],"qualityScoresReverseStrand":[{"type":"NumberWithFrequency","frequency":16,"number":40}],"readIndicesForwardStrand":[{"type":"NumberWithFrequency","frequency":1,"number":47},{"type":"NumberWithFrequency","frequency":1,"number":99}],"readIndicesReverseStrand":[{"type":"NumberWithFrequency","frequency":2,"number":6},{"type":"NumberWithFrequency","frequency":2,"number":7},{"type":"NumberWithFrequency","frequency":1,"number":12},{"type":"NumberWithFrequency","frequency":1,"number":14},{"type":"NumberWithFrequency","frequency":1,"number":19},{"type":"NumberWithFrequency","frequency":1,"number":40},{"type":"NumberWithFrequency","frequency":1,"number":46},{"type":"NumberWithFrequency","frequency":1,"number":48},{"type":"NumberWithFrequency","frequency":1,"number":64},{"type":"NumberWithFrequency","frequency":5,"number":69}],"readMappingQualityForwardStrand":[{"type":"NumberWithFrequency","frequency":2,"number":60}],"readMappingQualityReverseStrand":[{"type":"NumberWithFrequency","frequency":16,"number":60}],"numVariationsInReads":[{"type":"NumberWithFrequency","frequency":7,"number":0},{"type":"NumberWithFrequency","frequency":9,"number":1},{"type":"NumberWithFrequency","frequency":2,"number":2}],"insertSizes":[{"type":"NumberWithFrequency","frequency":1,"number":-488},{"type":"NumberWithFrequency","frequency":1,"number":-481},{"type":"NumberWithFrequency","frequency":1,"number":-469},{"type":"NumberWithFrequency","frequency":1,"number":-450},{"type":"NumberWithFrequency","frequency":1,"number":-429},{"type":"NumberWithFrequency","frequency":1,"number":-419},{"type":"NumberWithFrequency","frequency":1,"number":-401},{"type":"NumberWithFrequency","frequency":1,"number":-359},{"type":"NumberWithFrequency","frequency":1,"number":-349},{"type":"NumberWithFrequency","frequency":1,"number":-337},{"type":"NumberWithFrequency","frequency":1,"number":-310},{"type":"NumberWithFrequency","frequency":1,"number":-301},{"type":"NumberWithFrequency","frequency":1,"number":-294},{"type":"NumberWithFrequency","frequency":1,"number":-292},{"type":"NumberWithFrequency","frequency":2,"number":0},{"type":"NumberWithFrequency","frequency":1,"number":383},{"type":"NumberWithFrequency","frequency":1,"number":410}],"targetAlignedLengths":[{"type":"NumberWithFrequency","frequency":2,"number":79},{"type":"NumberWithFrequency","frequency":4,"number":108},{"type":"NumberWithFrequency","frequency":6,"number":109},{"type":"NumberWithFrequency","frequency":4,"number":114},{"type":"NumberWithFrequency","frequency":2,"number":115},{"type":"NumberWithFrequency","frequency":2,"number":116},{"type":"NumberWithFrequency","frequency":2,"number":121},{"type":"NumberWithFrequency","frequency":2,"number":142},{"type":"NumberWithFrequency","frequency":2,"number":148},{"type":"NumberWithFrequency","frequency":2,"number":150},{"type":"NumberWithFrequency","frequency":4,"number":151},{"type":"NumberWithFrequency","frequency":4,"number":171}],"queryAlignedLengths":[{"type":"NumberWithFrequency","frequency":1,"number":78},{"type":"NumberWithFrequency","frequency":2,"number":108},{"type":"NumberWithFrequency","frequency":3,"number":109},{"type":"NumberWithFrequency","frequency":2,"number":114},{"type":"NumberWithFrequency","frequency":1,"number":115},{"type":"NumberWithFrequency","frequency":1,"number":116},{"type":"NumberWithFrequency","frequency":1,"number":121},{"type":"NumberWithFrequency","frequency":1,"number":142},{"type":"NumberWithFrequency","frequency":1,"number":148},{"type":"NumberWithFrequency","frequency":1,"number":150},{"type":"NumberWithFrequency","frequency":2,"number":151},{"type":"NumberWithFrequency","frequency":2,"number":171}],"queryPositions":[{"type":"NumberWithFrequency","frequency":18,"number":0}],"pairFlags":[{"type":"NumberWithFrequency","frequency":2,"number":16},{"type":"NumberWithFrequency","frequency":8,"number":83},{"type":"NumberWithFrequency","frequency":2,"number":99},{"type":"NumberWithFrequency","frequency":6,"number":147}],"distancesToReadVariationsForwardStrand":[{"type":"NumberWithFrequency","frequency":1,"number":-23}],"distancesToReadVariationsReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":-98},{"type":"NumberWithFrequency","frequency":1,"number":-84},{"type":"NumberWithFrequency","frequency":1,"number":-71},{"type":"NumberWithFrequency","frequency":1,"number":-69},{"type":"NumberWithFrequency","frequency":1,"number":-57},{"type":"NumberWithFrequency","frequency":1,"number":-54},{"type":"NumberWithFrequency","frequency":4,"number":-40},{"type":"NumberWithFrequency","frequency":1,"number":-8},{"type":"NumberWithFrequency","frequency":1,"number":4}],"distanceToStartOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":9},{"type":"NumberWithFrequency","frequency":1,"number":40},{"type":"NumberWithFrequency","frequency":1,"number":45},{"type":"NumberWithFrequency","frequency":1,"number":47},{"type":"NumberWithFrequency","frequency":1,"number":87},{"type":"NumberWithFrequency","frequency":1,"number":99},{"type":"NumberWithFrequency","frequency":12,"number":102}],"distanceToEndOfRead":[{"type":"NumberWithFrequency","frequency":2,"number":6},{"type":"NumberWithFrequency","frequency":2,"number":7},{"type":"NumberWithFrequency","frequency":1,"number":12},{"type":"NumberWithFrequency","frequency":1,"number":14},{"type":"NumberWithFrequency","frequency":1,"number":19},{"type":"NumberWithFrequency","frequency":1,"number":40},{"type":"NumberWithFrequency","frequency":1,"number":46},{"type":"NumberWithFrequency","frequency":1,"number":48},{"type":"NumberWithFrequency","frequency":1,"number":52},{"type":"NumberWithFrequency","frequency":1,"number":64},{"type":"NumberWithFrequency","frequency":1,"number":68},{"type":"NumberWithFrequency","frequency":5,"number":69}]},{"type":"CountInfo","matchesReference":false,"isCalled":true,"isIndel":true,"fromSequence":"C-TTTTTTTTTTTTAT","toSequence":"CTTTTTTTTTTTTTAT","genotypeCountForwardStrand":6,"genotypeCountReverseStrand":10,"gobyGenotypeIndex":5,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[{"type":"NumberWithFrequency","frequency":1,"number":14},{"type":"NumberWithFrequency","frequency":1,"number":47},{"type":"NumberWithFrequency","frequency":1,"number":60},{"type":"NumberWithFrequency","frequency":1,"number":69},{"type":"NumberWithFrequency","frequency":1,"number":93},{"type":"NumberWithFrequency","frequency":1,"number":104}],"readIndicesReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":19},{"type":"NumberWithFrequency","frequency":1,"number":30},{"type":"NumberWithFrequency","frequency":1,"number":34},{"type":"NumberWithFrequency","frequency":1,"number":48},{"type":"NumberWithFrequency","frequency":1,"number":55},{"type":"NumberWithFrequency","frequency":1,"number":69}],"readMappingQualityForwardStrand":[{"type":"NumberWithFrequency","frequency":6,"number":60}],"readMappingQualityReverseStrand":[{"type":"NumberWithFrequency","frequency":10,"number":60}],"numVariationsInReads":[{"type":"NumberWithFrequency","frequency":2,"number":1},{"type":"NumberWithFrequency","frequency":7,"number":2},{"type":"NumberWithFrequency","frequency":5,"number":3},{"type":"NumberWithFrequency","frequency":1,"number":4},{"type":"NumberWithFrequency","frequency":1,"number":5}],"insertSizes":[{"type":"NumberWithFrequency","frequency":1,"number":-427},{"type":"NumberWithFrequency","frequency":1,"number":-404},{"type":"NumberWithFrequency","frequency":1,"number":-382},{"type":"NumberWithFrequency","frequency":1,"number":-361},{"type":"NumberWithFrequency","frequency":1,"number":-356},{"type":"NumberWithFrequency","frequency":1,"number":-301},{"type":"NumberWithFrequency","frequency":4,"number":0},{"type":"NumberWithFrequency","frequency":1,"number":285},{"type":"NumberWithFrequency","frequency":1,"number":291},{"type":"NumberWithFrequency","frequency":1,"number":339},{"type":"NumberWithFrequency","frequency":1,"number":397},{"type":"NumberWithFrequency","frequency":1,"number":485},{"type":"NumberWithFrequency","frequency":1,"number":612}],"targetAlignedLengths":[{"type":"NumberWithFrequency","frequency":2,"number":81},{"type":"NumberWithFrequency","frequency":2,"number":105},{"type":"NumberWithFrequency","frequency":2,"number":114},{"type":"NumberWithFrequency","frequency":2,"number":121},{"type":"NumberWithFrequency","frequency":2,"number":127},{"type":"NumberWithFrequency","frequency":2,"number":132},{"type":"NumberWithFrequency","frequency":4,"number":136},{"type":"NumberWithFrequency","frequency":2,"number":144},{"type":"NumberWithFrequency","frequency":6,"number":150},{"type":"NumberWithFrequency","frequency":8,"number":171}],"queryAlignedLengths":[{"type":"NumberWithFrequency","frequency":1,"number":82},{"type":"NumberWithFrequency","frequency":1,"number":106},{"type":"NumberWithFrequency","frequency":1,"number":115},{"type":"NumberWithFrequency","frequency":1,"number":122},{"type":"NumberWithFrequency","frequency":1,"number":128},{"type":"NumberWithFrequency","frequency":1,"number":133},{"type":"NumberWithFrequency","frequency":2,"number":137},{"type":"NumberWithFrequency","frequency":1,"number":145},{"type":"NumberWithFrequency","frequency":3,"number":151},{"type":"NumberWithFrequency","frequency":4,"number":172}],"queryPositions":[{"type":"NumberWithFrequency","frequency":16,"number":0}],"pairFlags":[{"type":"NumberWithFrequency","frequency":4,"number":16},{"type":"NumberWithFrequency","frequency":3,"number":83},{"type":"NumberWithFrequency","frequency":5,"number":99},{"type":"NumberWithFrequency","frequency":3,"number":147},{"type":"NumberWithFrequency","frequency":1,"number":163}],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"C","toSequence":"A","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":0,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"C","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"C","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"C","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]}]}]}'
        count_mapper = MapCountInfo(device=device,count_dim=32)
        sample_mapper = MapSampleInfo(count_mapper=count_mapper, count_dim=32, sample_dim=64, num_counts=3, device=device)
        record_mapper = MapBaseInformation(sample_mapper=sample_mapper, sample_dim=64, num_samples=1, ploidy=2,
                                           extra_genotypes=2, device = device)
        batch_of_records = BatchOfRecords(record_mapper,device = device)
        import ujson
        executor = FoldExecutor(count_mapper=count_mapper, sample_mapper=sample_mapper,
                                record_mapper=record_mapper)


        fold = torchfold.Fold(executor)
        mapped_nodes = []

        record = ujson.loads(test_json)

        loaded = batch_of_records.preload([record])
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root", loaded))
        print(fold)
        mapped = fold.apply(executor, mapped_nodes)
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

if __name__ == '__main__':
    unittest.main()
