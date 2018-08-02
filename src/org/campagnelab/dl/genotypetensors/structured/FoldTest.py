import torch
import unittest

from torchfold import torchfold

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import MapCountInfo, MapSampleInfo, MapBaseInformation

cuda = torch.device('cuda')
cpu = torch.device('cpu')


class FoldExecutor:
    def __init__(self, map_counts):
        self.map_counts = map_counts

    def root_count_map_count(self, count_value):
        return self.map_counts.map_count.simple_forward(count_value)

    def root_count_reduce_count(self, *tensors):
        return self.map_counts.reduce_count.forward_flat_inputs(torch.cat(tensors,dim=1))

    def root_count_map_gobyGenotypeIndex(self, value):
        return self.map_counts.map_gobyGenotypeIndex.simple_forward(value)

    def root_count_map_boolean(self, value):
        return self.map_counts.map_boolean.loaded_forward(value)

    def root_count_map_sequence(self, value):
        return self.map_counts.map_sequence.simple_forward(value)

    def nwl_map_nwl(self, mapper, numbers, frequencies):
        mapped_frequencies = torch.cat([
            mapper.map_number.simple_forward(numbers),
            mapper.map_frequency.simple_forward(frequencies)], dim=1)

        return mapper.map_sequence(mapped_frequencies)

    def root_count_qualityScoresForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_base_qual, numbers, frequencies)

    def root_count_qualityScoresReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_base_qual, numbers, frequencies)

    def root_count_distanceToStartOfRead_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_distance_to, numbers, frequencies)

    def root_count_distanceToEndOfRead_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_distance_to, numbers, frequencies)

    def root_count_readIndicesReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_read_indices, numbers, frequencies)

    def root_count_readIndicesForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_read_indices, numbers, frequencies)


    def root_count_targetAlignedLengths_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_aligned_lengths, numbers, frequencies)

    def root_count_queryAlignedLengths_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_aligned_lengths, numbers, frequencies)

    def root_count_numVariationsInReads_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_num_var, numbers, frequencies)

    def root_count_readMappingQualityForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_mapping_qual, numbers, frequencies)

    def root_count_readMappingQualityReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.map_counts.frequency_list_mapper_mapping_qual, numbers, frequencies)


class PreloadTestCase(unittest.TestCase):
    def test_preload_count(self):
        device = cpu
        map_container = MapCountInfo(device=device)
        import ujson


        fold = torchfold.Fold()
        mapped_nodes = []

        record = ujson.loads(sbi_json_string)
        count = record['samples'][0]['counts'][0]
        loaded = map_container.preload(count)
        # move preloaded tensors to cuda:
        loaded.to(device)
        mapped_nodes.append(loaded.mapper.fold(fold, "root", loaded))
        excutor = FoldExecutor(map_container)
        print(fold)
        mapped = fold.apply(excutor, [mapped_nodes])
        self.assertIsNotNone(mapped)
        loaded.to(torch.device(cpu))
        print(mapped)

if __name__ == '__main__':
    unittest.main()
