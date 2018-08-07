import torch
import unittest

from torchfold import torchfold

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string, \
    FoldExecutor
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo, MapSampleInfo, MapBaseInformation, \
    MapNumberWithFrequencyList

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
        executor = FoldExecutor(count_mapper=count_mapper, sample_mapper=sample_mapper)


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


    def test_fold_sample(self):
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


if __name__ == '__main__':
    unittest.main()
