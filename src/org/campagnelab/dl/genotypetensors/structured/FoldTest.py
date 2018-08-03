import torch
import unittest

from torchfold import torchfold

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo, MapSampleInfo, MapBaseInformation

cuda = torch.device('cuda')
cpu = torch.device('cpu')


class FoldExecutor:
    def __init__(self, count_mapper, sample_mapper):
        self.count_mapper = count_mapper
        self.sample_mapper = sample_mapper

    def root_count_map_count(self, count_value):
        return self.count_mapper.map_count.simple_forward(count_value)

    def root_count_reduce_count(self, *tensors):
        return self.count_mapper.reduce_count.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_sample_reduce_count(self, *tensors):
        return self.sample_mapper.reduce_counts.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_count_map_gobyGenotypeIndex(self, value):
        return self.count_mapper.map_gobyGenotypeIndex.simple_forward(value)

    def root_count_map_boolean(self, value):
        return self.count_mapper.map_boolean.loaded_forward(value)

    def root_count_map_sequence(self, value):
        return self.count_mapper.map_sequence.simple_forward(value)

    def nwl_map_nwl(self, mapper, numbers, frequencies):
        mapped_frequencies = torch.cat([
            mapper.map_number.simple_forward(numbers),
            mapper.map_frequency.simple_forward(frequencies)], dim=1)

        return mapper.map_sequence(mapped_frequencies)

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
        excutor = FoldExecutor(count_mapper)
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

        fold = torchfold.Unfold(executor)
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


if __name__ == '__main__':
    unittest.main()
