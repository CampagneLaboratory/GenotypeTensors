from math import log2, log10

import torch
from torch.autograd import Variable
from torch.nn import Module

from org.campagnelab.dl.genotypetensors.structured.Models import Reduce, IntegerModel, map_Boolean, RNNOfList, \
    StructuredEmbedding, NoCache


class MapSequence(StructuredEmbedding):
    def __init__(self, mapped_base_dim=2, hidden_size=64, num_layers=1, bases=('A', 'C', 'T', 'G', '-', 'N')):
        super().__init__(embedding_size=hidden_size)
        self.map_sequence = RNNOfList(embedding_size=mapped_base_dim, hidden_size=hidden_size,
                                      num_layers=num_layers)
        max_base_index = 0
        for base in bases:
            max_base_index = max(ord(base[0]), max_base_index)
        self.map_bases = IntegerModel(distinct_numbers=(max_base_index + 1), embedding_size=mapped_base_dim)

    def forward(self, sequence_field, tensor_cache, cuda=None):
        return self.map_sequence(
            self.map_bases(list([ord(b) for b in sequence_field]), tensor_cache=tensor_cache, cuda=cuda), cuda)


class MapBaseInformation(Module):
    def __init__(self, sample_mapper, sample_dim, num_samples, sequence_output_dim=64):
        super().__init__()
        self.sample_mapper = sample_mapper
        mapped_base_dim = 2
        bases = ('A', 'C', 'T', 'G', '-', 'N')
        self.map_sequence = MapSequence(hidden_size=sequence_output_dim, bases=bases,
                                        mapped_base_dim=mapped_base_dim)
        self.reduce_samples = Reduce([sequence_output_dim] + [sequence_output_dim] + [sample_dim] * num_samples,
                                     encoding_output_dim=sample_dim)

        self.num_samples = num_samples

    def forward(self, input, tensor_cache, cuda=None):
        if cuda is None:
            cuda = next(self.parameters()).data.is_cuda

        return self.reduce_samples([self.map_sequence(input['referenceBase'], tensor_cache=tensor_cache, cuda=cuda)] +
                                   [self.map_sequence(input['genomicSequenceContext'], tensor_cache=tensor_cache,
                                                      cuda=cuda)] +
                                   [self.sample_mapper(sample, tensor_cache=tensor_cache, cuda=cuda) for sample in
                                    input['samples'][0:self.num_samples]], cuda)


class MapSampleInfo(Module):
    def __init__(self, count_mapper, count_dim, sample_dim, num_counts):
        super().__init__()
        self.count_mapper = count_mapper
        self.num_counts = num_counts
        self.reduce_counts = Reduce([count_dim] * num_counts, encoding_output_dim=sample_dim)

    def forward(self, input, tensor_cache, cuda=None):
        observed_counts = [count for count in input['counts'] if
                           (count['genotypeCountForwardStrand'] + count['genotypeCountReverseStrand']) > 0]
        return self.reduce_counts([self.count_mapper(count, tensor_cache=tensor_cache, cuda=cuda) for count in
                                   observed_counts[0:self.num_counts]],
                                  pad_missing=True, cuda=cuda)


class MapCountInfo(Module):
    def __init__(self, mapped_count_dim=5, count_dim=64, mapped_base_dim=6, mapped_genotype_index_dim=4):
        super().__init__()
        self.map_sequence = MapSequence(bases=['A', 'C', 'T', 'G', '-'], hidden_size=count_dim,
                                        mapped_base_dim=mapped_base_dim)
        self.map_gobyGenotypeIndex = IntegerModel(distinct_numbers=100, embedding_size=mapped_genotype_index_dim)

        self.map_count = IntegerModel(distinct_numbers=100000, embedding_size=mapped_count_dim)
        self.map_boolean = map_Boolean()

        self.frequency_list_mapper_base_qual = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_num_var = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_mapping_qual = MapNumberWithFrequencyList(distinct_numbers=100)
        self.frequency_list_mapper_distance_to = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_aligned_lengths = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_read_indices = MapNumberWithFrequencyList(distinct_numbers=1000)

        count_mappers = [self.map_gobyGenotypeIndex,
                         self.map_boolean,  # isIndel
                         self.map_boolean,  # matchesReference
                         self.map_sequence,
                         self.map_sequence,
                         self.map_count,
                         self.map_count]

        self.nf_names_mappers = [('qualityScoresForwardStrand', self.frequency_list_mapper_base_qual),
                                 ('qualityScoresReverseStrand', self.frequency_list_mapper_base_qual),
                                 ('distanceToStartOfRead', self.frequency_list_mapper_distance_to),
                                 ('distanceToEndOfRead', self.frequency_list_mapper_distance_to),  # OK
                                 ('readIndicesReverseStrand', self.frequency_list_mapper_read_indices),  # OK
                                 ('readIndicesForwardStrand', self.frequency_list_mapper_read_indices),  # OK
                                 # 'distancesToReadVariationsForwardStrand', #Wrong
                                 # 'distancesToReadVariationsReverseStrand', #Wrong
                                 ('targetAlignedLengths', self.frequency_list_mapper_aligned_lengths),
                                 ('queryAlignedLengths', self.frequency_list_mapper_aligned_lengths),  # OK
                                 ('numVariationsInReads', self.frequency_list_mapper_num_var),  # OK
                                 ('readMappingQualityForwardStrand', self.frequency_list_mapper_mapping_qual),  # OK
                                 ('readMappingQualityReverseStrand', self.frequency_list_mapper_mapping_qual)  # OK
                                 ]
        for nf_name, mapper in self.nf_names_mappers:
            count_mappers += [mapper]

        # below, [2+2+2] is for the booleans mapped with a function:
        self.reduce_count = Reduce([mapper.embedding_size for mapper in count_mappers], encoding_output_dim=count_dim)
        batched_count_mappers=[]
        self.reduce_batched = Reduce([self.map_gobyGenotypeIndex.embedding_size,
                                      self.map_count.embedding_size * 2,
                                      self.map_boolean.embedding_size * 2,
                                      self.map_sequence.embedding_size * 2, ], encoding_output_dim=count_dim)

    def forward(self, c, tensor_cache, cuda=None):
        mapped_gobyGenotypeIndex = self.map_gobyGenotypeIndex([c['gobyGenotypeIndex']], tensor_cache=tensor_cache,
                                                              cuda=cuda)
        # Do not map isCalled, it is a field that contains the truth and is used to calculate the label.

        mapped_isIndel = self.map_boolean(c['isIndel'], tensor_cache=tensor_cache, cuda=cuda)
        mapped_matchesReference = self.map_boolean(c['matchesReference'], tensor_cache=tensor_cache, cuda=cuda)

        mapped_from = self.map_sequence(c['fromSequence'], tensor_cache=tensor_cache, cuda=cuda)
        mapped_to = self.map_sequence(c['toSequence'], tensor_cache=tensor_cache, cuda=cuda)
        mapped_genotypeCountForwardStrand = self.map_count([c['genotypeCountForwardStrand']], tensor_cache=tensor_cache,
                                                           cuda=cuda)
        mapped_genotypeCountReverseStrand = self.map_count([c['genotypeCountReverseStrand']], tensor_cache=tensor_cache,
                                                           cuda=cuda)

        mapped = [mapped_gobyGenotypeIndex,
                  mapped_isIndel,
                  mapped_matchesReference,
                  mapped_from,
                  mapped_to,
                  mapped_genotypeCountForwardStrand,
                  mapped_genotypeCountReverseStrand]

        for nf_name, mapper in self.nf_names_mappers:
            if nf_name in c.keys():
                mapped += [mapper(c[nf_name],
                                  tensor_cache=tensor_cache,
                                  cuda=cuda, nf_name=nf_name)]
            else:
                mapped += [Variable(torch.zeros(1, mapper.embedding_size), requires_grad=True)]
        return self.reduce_count(mapped, cuda)

    def cat_inputs(self, mapper, list_of_values, tensor_cache=NoCache(), phase=0, cuda=False, direct_forward=False):
        mapper_id = id(mapper)
        results = {mapper_id: []}
        for value in list_of_values:
            if direct_forward:
                mapped = mapper(value, tensor_cache=tensor_cache, cuda=cuda)
            else:
                mapped = mapper.collect_inputs(value, tensor_cache=tensor_cache, phase=phase, cuda=cuda)

            results[mapper_id] += [mapped]

        return torch.cat(results[mapper_id], dim=1)

    def collect_inputs(self, c, phase=0, tensor_cache=NoCache(), cuda=None, batcher=None):
        if phase == 0:
            return {
                # the following tensors are batched:
                id(self.map_gobyGenotypeIndex): self.map_gobyGenotypeIndex.collect_inputs(
                    values=[c['gobyGenotypeIndex']], phase=phase, cuda=cuda),
                id(self.map_count): self.map_count.collect_inputs(
                    values=[c['genotypeCountForwardStrand'], c['genotypeCountReverseStrand']], phase=phase, cuda=cuda),
                id(self.map_boolean): self.map_boolean.collect_inputs(values=[c['isIndel'], c['matchesReference']],
                                                                      tensor_cache=tensor_cache, phase=phase,
                                                                      cuda=cuda),
                # the following tensors are not batched, but computed once per instance, right here with direct_forward=True:
                id(self.map_sequence): self.cat_inputs(self.map_sequence, [c['fromSequence'], c['toSequence']],
                                                       tensor_cache=tensor_cache, phase=phase, cuda=cuda,
                                                       direct_forward=True)
            }
        if phase == 1:
            count_index = c['count_index']
            # TODO: restrict the batched results to the current slice corresponding to this count index.
            # TODO: the index depends on how many items of each type were mapped for the example.
            current_results=batcher.get_batched_result(self)
            mapped_goby_genotype_indices = current_results[id(self.map_gobyGenotypeIndex)]
            mapped_counts = current_results[id(self.map_count)]
            mapped_booleans = current_results[id(self.map_boolean)]

            # mapped_sequences are not currently batchable, so we store the input from the prior phase:
            mapped_sequences = batcher.get_batched_input(mapper=self.map_sequence)

            all_mapped = [mapped_goby_genotype_indices, mapped_counts, mapped_booleans, mapped_sequences]
            return self.reduce_batched(all_mapped)


    def forward_batch(self, batcher, phase=0):
        if phase==0:
            return  {
                    id(self.map_gobyGenotypeIndex): self.map_gobyGenotypeIndex.forward_batch(batcher),
                    id(self.map_count): self.map_count.forward_batch(batcher),
                    id(self.map_boolean): self.map_boolean.forward_batch(batcher),
                    id(self.map_sequence):batcher.get_batched_input(mapper=self.map_sequence)
                }
        if phase==1:
            return {id(self): batcher.get_batched_input(self)}


class MapNumberWithFrequencyList(StructuredEmbedding):
    def __init__(self, distinct_numbers=-1, mapped_number_dim=4):
        mapped_frequency_dim = 3
        super().__init__(embedding_size=mapped_number_dim + mapped_frequency_dim)

        output_dim = mapped_number_dim + mapped_frequency_dim
        self.map_number = IntegerModel(distinct_numbers=distinct_numbers, embedding_size=mapped_number_dim)
        # self.map_frequency = Variable(torch.FloatTensor([[]]))
        # IntegerModel(distinct_numbers=distinct_frequencies, embedding_size=mapped_frequency_dim)

        self.map_sequence = RNNOfList(embedding_size=mapped_number_dim + mapped_frequency_dim, hidden_size=output_dim,
                                      num_layers=1)

    def map_frequency(self, value):
        """We use three floats to represent each frequency:"""
        return Variable(torch.FloatTensor([log10(value) - log10(10), log2(value) - log2(10), value / 10]),
                        requires_grad=True)

    def forward(self, nwf_list, tensor_cache, cuda=None, nf_name="unknown"):

        if len(nwf_list) > 0:
            mapped_frequencies = [torch.cat([
                self.map_number([nwf['number']], tensor_cache, cuda).squeeze(),
                self.map_frequency(nwf['frequency'])], dim=0) for nwf in nwf_list]
            mapped_frequencies = torch.stack(mapped_frequencies, dim=0)
            return self.map_sequence(mapped_frequencies, cuda=cuda)
        else:
            return Variable(torch.zeros(1, self.embedding_size), requires_grad=True)


def configure_mappers(ploidy, extra_genotypes, num_samples, sample_dim=64, count_dim=64):
    """Return a tuple with two elements:
    mapper-dictionary: key is name of message type. value is function to map the message.
    all-modules: list of modules that implement mapping. """

    num_counts = ploidy + extra_genotypes

    map_CountInfo = MapCountInfo(mapped_count_dim=5, count_dim=count_dim, mapped_base_dim=6,
                                 mapped_genotype_index_dim=2)
    map_SampleInfo = MapSampleInfo(count_mapper=map_CountInfo, num_counts=num_counts, count_dim=count_dim,
                                   sample_dim=sample_dim)
    map_SbiRecords = MapBaseInformation(sample_mapper=map_SampleInfo, num_samples=num_samples, sample_dim=sample_dim,
                                        sequence_output_dim=count_dim)

    sbi_mappers = {"BaseInformation": map_SbiRecords,
                   "SampleInfo": map_SampleInfo,
                   "CountInfo": map_CountInfo
                   }
    return sbi_mappers, [map_SbiRecords, map_SampleInfo, map_CountInfo]
