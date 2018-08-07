import torch
import unittest

from torchfold import torchfold

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo, MapSampleInfo, MapBaseInformation, \
    MapNumberWithFrequencyList

cuda = torch.device('cuda')
cpu = torch.device('cpu')


class FoldExecutor:
    def __init__(self, count_mapper, sample_mapper, record_mapper):
        self.count_mapper = count_mapper
        self.sample_mapper = sample_mapper
        self.record_mapper=record_mapper

    def root_count_map_count(self, count_value):
        return self.count_mapper.map_count.simple_forward(count_value)

    def root_genomic_context_sequence(self, preloaded):
        return self.count_mapper.map_sequence.simple_forward(preloaded)

    def root_from_sequence_sequence(self, preloaded):
        return self.count_mapper.map_sequence.simple_forward(preloaded)

    def root_ref_base_sequence(self, preloaded):
        return self.count_mapper.map_sequence.simple_forward(preloaded)

    def root_count_reduce_count(self, *tensors):
        return self.count_mapper.reduce_count.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_sample_reduce_count(self, *tensors):
        return self.sample_mapper.reduce_counts.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_record_reduce_count(self, *tensors):
        return self.record_mapper.reduce_samples.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_count_map_gobyGenotypeIndex(self, value):
        return self.count_mapper.map_gobyGenotypeIndex.simple_forward(value)

    def root_count_map_boolean(self, value):
        return self.count_mapper.map_boolean.loaded_forward(value)

    def root_count_map_sequence(self, value):
        return self.count_mapper.map_sequence.simple_forward(value)

    def nwl_map_nwl(self, mapper, numbers, frequencies):
        batch_size=numbers.size(0)
        mapped_numbers = mapper.map_number.simple_forward(numbers).squeeze()
        mapped_frequencies = mapper.map_frequency.simple_forward(frequencies).squeeze()
        last_dim=len(mapped_numbers.size())-1
        concatenated = torch.cat([
            mapped_numbers,
            mapped_frequencies], dim=last_dim)
        embedding_size=mapper.map_number.embedding_size+mapper.map_frequency.embedding_size
        mapped= mapper.map_sequence.simple_forward(concatenated.view(batch_size,-1,embedding_size))
        return mapped.view(batch_size,embedding_size)

    def root_count_qualityScoresForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_base_qual, numbers, frequencies)

    def root_count_qualityScoresReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_base_qual, numbers, frequencies)

    def root_count_distanceToStartOfRead_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_distance_to, numbers, frequencies)

    def root_count_distanceToEndOfRead_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_distance_to, numbers, frequencies)

    def root_count_readIndicesReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_read_indices, numbers, frequencies)

    def root_count_readIndicesForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_read_indices, numbers, frequencies)


    def root_count_targetAlignedLengths_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_aligned_lengths, numbers, frequencies)

    def root_count_queryAlignedLengths_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_aligned_lengths, numbers, frequencies)

    def root_count_numVariationsInReads_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_num_var, numbers, frequencies)

    def root_count_readMappingQualityForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_mapping_qual, numbers, frequencies)

    def root_count_readMappingQualityReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_mapping_qual, numbers, frequencies)

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
        excutor = FoldExecutor(count_mapper=count_mapper,sample_mapper=None)
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
        excutor = FoldExecutor(count_mapper=count_mapper,sample_mapper=None)
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
        excutor = FoldExecutor(count_mapper,sample_mapper=None)
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
