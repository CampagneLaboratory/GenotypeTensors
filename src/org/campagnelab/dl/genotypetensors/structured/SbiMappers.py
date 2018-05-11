from math import log2, log10

import torch
from torch.autograd import Variable
from torch.nn import Module

from org.campagnelab.dl.genotypetensors.structured.Models import Reduce, IntegerModel, map_Boolean, RNNOfList, \
    StructuredEmbedding, NoCache, MeanOfList


def store_indices_in_message(mapper, message, indices):
    mapper_id = id(mapper)
    if mapper_id not in message['indices']:
        message['indices'][mapper_id] = []
    message['indices'][mapper_id] += indices


def get_indices_in_message(mapper, message):
    return message['indices'][id(mapper)]

use_mean_to_map_nwf=False

class MapSequence(StructuredEmbedding):
    def __init__(self, mapped_base_dim=2, hidden_size=64, num_layers=1, bases=('A', 'C', 'T', 'G', '-', 'N')):
        super().__init__(embedding_size=hidden_size)
        self.map_sequence = RNNOfList(embedding_size=mapped_base_dim, hidden_size=hidden_size,
                                      num_layers=num_layers)
        self.base_to_index={}
        for base_index, base in enumerate(bases):
            self.base_to_index[base[0]]=base_index

        self.map_bases = IntegerModel(distinct_numbers=len(self.base_to_index), embedding_size=mapped_base_dim)

    def forward(self, sequence_field, tensor_cache=NoCache(), cuda=None):
        return self.map_sequence(
            self.map_bases(list([self.base_to_index[b] for b in sequence_field]), tensor_cache=tensor_cache, cuda=cuda), cuda)


class MapBaseInformation(Module):
    def __init__(self, sample_mapper, sample_dim, num_samples, sequence_output_dim=64, ploidy=2, extra_genotypes=2):
        super().__init__()
        self.sample_mapper = sample_mapper
        mapped_base_dim = 2
        bases = ('A', 'C', 'T', 'G', '-', 'N')
        self.map_sequence = MapSequence(hidden_size=sequence_output_dim, bases=bases,
                                        mapped_base_dim=mapped_base_dim)
        self.reduce_samples = Reduce([sequence_output_dim] + [sequence_output_dim] + [sample_dim] * num_samples,
                                     encoding_output_dim=sample_dim)

        self.num_samples = num_samples
        self.ploidy = ploidy
        self.extra_genotypes = extra_genotypes

    def forward(self, input, tensor_cache=NoCache(), cuda=None):
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
        self.count_dim = count_dim
        self.reduce_counts = Reduce([count_dim] * num_counts, encoding_output_dim=sample_dim)

    def forward(self, input, tensor_cache, cuda=None):
        observed_counts = self.get_observed_counts(input)
        return self.reduce_counts([self.count_mapper(count, tensor_cache=tensor_cache, cuda=cuda) for count in
                                   observed_counts[0:self.num_counts]],
                                  pad_missing=True, cuda=cuda)

    def get_observed_counts(self, input):
        return [count for count in input['counts'] if
                (count['genotypeCountForwardStrand'] + count['genotypeCountReverseStrand']) > 0]

    def collect_inputs(self, sample, phase=0, tensor_cache=NoCache(), cuda=None, batcher=None):
        """Collect input data for all counts in this sample. """
        if 'indices' not in sample:
            sample['indices'] = {}
        observed_counts = self.get_observed_counts(sample)[0:self.num_counts]
        if phase < 2:

            for count in observed_counts:
                store_indices_in_message(mapper=self.count_mapper, message=count, indices=
                self.count_mapper.collect_inputs(count, tensor_cache=tensor_cache,
                                                 cuda=cuda, phase=phase, batcher=batcher))

            return []

        if phase == 2:
            list_mapped_counts = []
            for count in observed_counts:
                mapped_count = batcher.get_forward_for_example(self.count_mapper,
                                                               example_indices=count['indices'][id(self.count_mapper)])
                list_mapped_counts += [mapped_count]
            while len(list_mapped_counts) < self.num_counts:
                # pad the list with zeros:
                variable = Variable(torch.zeros(*list_mapped_counts[0].size()), requires_grad=True)
                if cuda:
                    variable= variable.cuda(async=True)
                list_mapped_counts += [variable]
            cat_list_mapped_counts = torch.cat(list_mapped_counts, dim=-1)

            store_indices_in_message(mapper=self.reduce_counts, message=sample,
                                     indices=self.reduce_counts.collect_inputs(cat_list_mapped_counts,
                                                                 tensor_cache=tensor_cache,
                                                                 cuda=cuda,
                                                                 phase=phase,
                                                                 batcher=batcher))
            return []

    def forward_batch(self, batcher, phase=0):
        """delegate to counts for phases 0 to 1, then reduce the counts to produce the sample."""
        if phase < 2:
            return self.count_mapper.forward_batch(batcher=batcher, phase=phase)
        if phase == 2:
            my_counts_as_list = batcher.get_batched_input(self.reduce_counts)

            batched_counts = self.reduce_counts.forward_flat_inputs(my_counts_as_list)
            batcher.store_batched_result(self, batched_counts)
            return batched_counts


class MapCountInfo(StructuredEmbedding):
    def __init__(self, mapped_count_dim=5, count_dim=64, mapped_base_dim=2, mapped_genotype_index_dim=4):
        super().__init__(count_dim)
        self.map_sequence = MapSequence(hidden_size=count_dim,
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
                                # ('distanceToStartOfRead', self.frequency_list_mapper_distance_to),
                                # ('distanceToEndOfRead', self.frequency_list_mapper_distance_to),  # OK
                                # ('readIndicesReverseStrand', self.frequency_list_mapper_read_indices),  # OK
                                # ('readIndicesForwardStrand', self.frequency_list_mapper_read_indices),  # OK
                                 # 'distancesToReadVariationsForwardStrand', #Wrong
                                 # 'distancesToReadVariationsReverseStrand', #Wrong
                                # ('targetAlignedLengths', self.frequency_list_mapper_aligned_lengths),
                                # ('queryAlignedLengths', self.frequency_list_mapper_aligned_lengths),  # OK
                                # ('numVariationsInReads', self.frequency_list_mapper_num_var),  # OK
                                # ('readMappingQualityForwardStrand', self.frequency_list_mapper_mapping_qual),  # OK
                                # ('readMappingQualityReverseStrand', self.frequency_list_mapper_mapping_qual)  # OK
                                 ]
        for nf_name, mapper in self.nf_names_mappers:
            count_mappers += [mapper]

        # below, [2+2+2] is for the booleans mapped with a function:
        self.reduce_count = Reduce([mapper.embedding_size for mapper in count_mappers], encoding_output_dim=count_dim)
        batched_count_mappers = []
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
                variable = Variable(torch.zeros(1, mapper.embedding_size), requires_grad=True)
                if cuda:
                    variable=variable.cuda(async=True)
                mapped += [variable]
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
            c['indices'] = {}
            # the following tensors are batched:
            store_indices_in_message(mapper=self.map_gobyGenotypeIndex, message=c,
                                     indices=self.map_gobyGenotypeIndex.collect_inputs(
                                         values=[c['gobyGenotypeIndex']], phase=phase, cuda=cuda, batcher=batcher))

            store_indices_in_message(mapper=self.map_count, message=c, indices=self.map_count.collect_inputs(
                values=[c['genotypeCountForwardStrand'], c['genotypeCountReverseStrand']], phase=phase, cuda=cuda,
                batcher=batcher))

            store_indices_in_message(mapper=self.map_boolean, message=c, indices=self.map_boolean.collect_inputs(
                values=[c['isIndel'], c['matchesReference']],
                tensor_cache=tensor_cache, phase=phase,
                cuda=cuda, batcher=batcher))

            c['mapped-not-batched'] = {}
            # the following tensors are not batched, but computed once per instance, right here with direct_forward=True:
            c['mapped-not-batched'][id(self.map_sequence)] = self.cat_inputs(self.map_sequence,
                                                                             [c['fromSequence'], c['toSequence']],
                                                                             tensor_cache=tensor_cache, phase=phase,
                                                                             cuda=cuda,
                                                                             direct_forward=True)
            return []

        if phase == 1:
            mapped_goby_genotype_indices = batcher.get_forward_for_example(mapper=self.map_gobyGenotypeIndex,
                                                                           message=c).view(1, -1)

            mapped_counts = batcher.get_forward_for_example(mapper=self.map_count, message=c).view(1, -1)

            mapped_booleans = batcher.get_forward_for_example(mapper=self.map_boolean, message=c).view(1, -1)

            # mapped_sequences are not currently batchable, so we get the input from the prior phase:
            mapped_sequences = c['mapped-not-batched'][id(self.map_sequence)].view(1, -1)

            all_mapped = [mapped_goby_genotype_indices, mapped_counts, mapped_booleans, mapped_sequences]

            return batcher.store_inputs(mapper=self, inputs=self.reduce_batched(all_mapped))

    def forward_batch(self, batcher, phase=0):
        if phase == 0:
            # calculate the forward on the batches:
            self.map_gobyGenotypeIndex.forward_batch(batcher)
            self.map_count.forward_batch(batcher)
            self.map_boolean.forward_batch(batcher)
            return None
        if phase == 1:
            batched_input = batcher.get_batched_input(mapper=self)
            batcher.store_batched_result(self, batched_result=batched_input)
            return batched_input


class MapNumberWithFrequencyList(StructuredEmbedding):
    def __init__(self, distinct_numbers=-1, mapped_number_dim=4):
        mapped_frequency_dim = 3
        super().__init__(embedding_size=mapped_number_dim + mapped_frequency_dim)

        output_dim = mapped_number_dim + mapped_frequency_dim
        self.map_number = IntegerModel(distinct_numbers=distinct_numbers, embedding_size=mapped_number_dim)
        # self.map_frequency = Variable(torch.FloatTensor([[]]))
        # IntegerModel(distinct_numbers=distinct_frequencies, embedding_size=mapped_frequency_dim)
        if use_mean_to_map_nwf:
            self.mean_sequence=MeanOfList()
        else:
            self.map_sequence = RNNOfList(embedding_size=mapped_number_dim + mapped_frequency_dim, hidden_size=output_dim,
                                          num_layers=1)

    def map_frequency(self, value,cuda=False):
        """We use three floats to represent each frequency:"""
        variable = Variable(torch.FloatTensor([log10(value) - log10(10), log2(value) - log2(10), value / 10]),
                            requires_grad=True)
        if cuda:
            variable=variable.cuda(async=True)
        return variable

    def forward(self, nwf_list, tensor_cache, cuda=None, nf_name="unknown"):

        if len(nwf_list) > 0:
            mapped_frequencies = [torch.cat([
                self.map_number([nwf['number']], tensor_cache, cuda).squeeze(),
                self.map_frequency(nwf['frequency'],cuda)], dim=0) for nwf in nwf_list]
            mapped_frequencies = torch.stack(mapped_frequencies, dim=0)
            if use_mean_to_map_nwf:
                return self.mean_sequence(mapped_frequencies,cuda=cuda)
            else:
                return self.map_sequence(mapped_frequencies, cuda=cuda)
        else:
            variable= Variable(torch.zeros(1, self.embedding_size), requires_grad=True)
            if cuda:
                variable=variable.cuda(async=True)
            return variable


def configure_mappers(ploidy, extra_genotypes, num_samples, sample_dim=64, count_dim=64):
    """Return a tuple with two elements:
    mapper-dictionary: key is name of message type. value is function to map the message.
    all-modules: list of modules that implement mapping. """

    num_counts = ploidy + extra_genotypes

    map_CountInfo = MapCountInfo(mapped_count_dim=5, count_dim=count_dim, mapped_base_dim=2,
                                 mapped_genotype_index_dim=2)
    map_SampleInfo = MapSampleInfo(count_mapper=map_CountInfo, num_counts=num_counts, count_dim=count_dim,
                                   sample_dim=sample_dim)
    map_SbiRecords = MapBaseInformation(sample_mapper=map_SampleInfo, num_samples=num_samples, sample_dim=sample_dim,
                                        sequence_output_dim=count_dim, ploidy=ploidy, extra_genotypes=extra_genotypes)

    sbi_mappers = {"BaseInformation": map_SbiRecords,
                   "SampleInfo": map_SampleInfo,
                   "CountInfo": map_CountInfo
                   }
    return sbi_mappers, [map_SbiRecords, map_SampleInfo, map_CountInfo]
