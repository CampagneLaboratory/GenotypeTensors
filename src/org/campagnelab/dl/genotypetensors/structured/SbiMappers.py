from math import log2, log10, log

import torch
from torch.autograd import Variable
from torch.nn import Module

from org.campagnelab.dl.genotypetensors.structured.Models import Reduce, IntegerModel, map_Boolean, RNNOfList, \
    StructuredEmbedding,  MeanOfList


def store_indices_in_message(mapper, message, indices):
    mapper_id = id(mapper)
    if mapper_id not in message['indices']:
        message['indices'][mapper_id] = []
    message['indices'][mapper_id] += indices


def get_indices_in_message(mapper, message):
    return message['indices'][id(mapper)]


use_mean_to_map_nwf = True


class MapSequence(StructuredEmbedding):
    def __init__(self, mapped_base_dim=2, hidden_size=64, num_layers=1, bases=('A', 'C', 'T', 'G', '-', 'N'),
                 device=None):
        super().__init__(embedding_size=hidden_size, device=device)
        self.map_sequence = RNNOfList(embedding_size=mapped_base_dim, hidden_size=hidden_size,
                                      num_layers=num_layers, device=device)
        self.base_to_index = {}
        for base_index, base in enumerate(bases):
            self.base_to_index[base[0]] = base_index

        self.map_bases = IntegerModel(distinct_numbers=len(self.base_to_index), embedding_size=mapped_base_dim,
                                      device=device)
        self.to(self.device)

    def forward(self, sequence_field):
        return self.map_sequence(
            self.map_bases(list([self.base_to_index[b] for b in sequence_field]))
            )


class MapBaseInformation(StructuredEmbedding):
    def __init__(self, sample_mapper, sample_dim, num_samples, sequence_output_dim=64, ploidy=2, extra_genotypes=2,
                 device=torch.device('cpu')):
        super().__init__(sample_dim, device)
        self.sample_mapper = sample_mapper

        # We reuse the same sequence mapper as the one used in MapCountInfo:
        self.map_sequence = sample_mapper.count_mapper.map_sequence
        self.reduce_samples = Reduce(
            [sequence_output_dim] + [sequence_output_dim] + [sample_dim + sequence_output_dim] * num_samples,
            encoding_output_dim=sample_dim, device=device)

        self.num_samples = num_samples
        self.ploidy = ploidy
        self.extra_genotypes = extra_genotypes

    def forward(self, input):

        return self.reduce_samples(
            [self.map_sequence(input['referenceBase'])] +
            [self.map_sequence(input['genomicSequenceContext'])] +
            # each sample has a unique from (same across all counts), which gets mapped and concatenated
            # with the sample mapped reduction:
            [torch.cat([self.sample_mapper(sample),
                        self.map_sequence(sample['counts'][0]['fromSequence'])], dim=1)
             for sample in input['samples'][0:self.num_samples]])


class MapSampleInfo(Module):
    def __init__(self, count_mapper, count_dim, sample_dim, num_counts, device=None):
        super().__init__()
        self.count_mapper = count_mapper
        self.num_counts = num_counts
        self.count_dim = count_dim
        self.reduce_counts = Reduce([count_dim] * num_counts, encoding_output_dim=sample_dim, device=device)

    def forward(self, input):
        observed_counts = self.get_observed_counts(input)
        return self.reduce_counts([self.count_mapper(count) for count in
                                   observed_counts[0:self.num_counts]],
                                  pad_missing=True)

    def get_observed_counts(self, input):
        return [count for count in input['counts'] if
                (count['genotypeCountForwardStrand'] + count['genotypeCountReverseStrand']) > 0]


class MapCountInfo(StructuredEmbedding):
    def __init__(self, mapped_count_dim=5, count_dim=64, mapped_base_dim=2, mapped_genotype_index_dim=4, device=None):
        super().__init__(count_dim, device)
        self.map_sequence = MapSequence(hidden_size=count_dim,
                                        mapped_base_dim=mapped_base_dim, device=device)
        self.map_gobyGenotypeIndex = IntegerModel(distinct_numbers=100, embedding_size=mapped_genotype_index_dim,
                                                  device=device)

        self.map_count = IntegerModel(distinct_numbers=100000, embedding_size=mapped_count_dim, device=device)
        self.map_boolean = map_Boolean(device=device)

        self.frequency_list_mapper_base_qual = MapNumberWithFrequencyList(distinct_numbers=1000, device=device)
        self.frequency_list_mapper_num_var = MapNumberWithFrequencyList(distinct_numbers=1000, device=device)
        self.frequency_list_mapper_mapping_qual = MapNumberWithFrequencyList(distinct_numbers=100, device=device)
        self.frequency_list_mapper_distance_to = MapNumberWithFrequencyList(distinct_numbers=1000, device=device)
        self.frequency_list_mapper_aligned_lengths = MapNumberWithFrequencyList(distinct_numbers=1000,
                                                                                device=device)
        self.frequency_list_mapper_read_indices = MapNumberWithFrequencyList(distinct_numbers=1000, device=device)

        count_mappers = [self.map_gobyGenotypeIndex,
                         self.map_boolean,  # isIndel
                         self.map_boolean,  # matchesReference
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
        self.reduce_count = Reduce([mapper.embedding_size for mapper in count_mappers], encoding_output_dim=count_dim,
                                   device=device)
        batched_count_mappers = []
        self.reduce_batched = Reduce([self.map_gobyGenotypeIndex.embedding_size,
                                      self.map_count.embedding_size * 2,
                                      self.map_boolean.embedding_size * 2,
                                      self.map_sequence.embedding_size * 2, ], encoding_output_dim=count_dim,
                                     device=device)

    def forward(self, c):
        mapped_gobyGenotypeIndex = self.map_gobyGenotypeIndex([c['gobyGenotypeIndex']])
        # Do not map isCalled, it is a field that contains the truth and is used to calculate the label.

        mapped_isIndel = self.map_boolean(c['isIndel'])
        mapped_matchesReference = self.map_boolean(c['matchesReference'])

        # NB: fromSequence was mapped at the level of BaseInformation.
        mapped_to = self.map_sequence(c['toSequence'])
        mapped_genotypeCountForwardStrand = self.map_count([c['genotypeCountForwardStrand']])
        mapped_genotypeCountReverseStrand = self.map_count([c['genotypeCountReverseStrand']])

        mapped = [mapped_gobyGenotypeIndex,
                  mapped_isIndel,
                  mapped_matchesReference,
                  mapped_to,
                  mapped_genotypeCountForwardStrand,
                  mapped_genotypeCountReverseStrand]

        for nf_name, mapper in self.nf_names_mappers:
            if nf_name in c.keys():
                mapped += [mapper(c[nf_name], nf_name=nf_name)]
            else:
                variable = torch.zeros(1, mapper.embedding_size, requires_grad=True)
                variable=variable.to(self.device)
                mapped += [variable]
        return self.reduce_count(mapped)




class FrequencyMapper(StructuredEmbedding):
    def __init__(self, device=None):
        super().__init__(3, device=device)
        self.LOG10 = log(10)
        self.LOG2 = log(2)
        self.epsilon = 1E-5

    def convert_list_of_floats(self, values):
        """This method accepts a list of floats."""
        x = torch.Tensor(values).view(-1, 1)
        if hasattr(self, 'epsilon'):
            x = x + self.epsilon

        return x.to(self.device)

    def forward(self, x):
        """We use two floats to represent each number (the natural log of the number and the number divided by 10). The input must be a Float tensor of dimension:
        (num-elements in list, 1), or a list of float values.
        """
        if not torch.is_tensor(x):
            x = self.convert_list_of_floats(x)

        x = torch.cat([torch.log(x) / self.LOG10, torch.log(x) / self.LOG2, x / 10.0], dim=1)

        return x


class MapNumberWithFrequencyList(StructuredEmbedding):
    def __init__(self, distinct_numbers=-1, mapped_number_dim=4, device=None):
        mapped_frequency_dim = 3
        super().__init__(embedding_size=mapped_number_dim + mapped_frequency_dim, device=device)

        output_dim = mapped_number_dim + mapped_frequency_dim
        self.map_number = IntegerModel(distinct_numbers=distinct_numbers, embedding_size=mapped_number_dim,
                                       device=self.device)
        self.map_frequency = FrequencyMapper(device=self.device)
        # self.map_frequency = Variable(torch.FloatTensor([[]]))
        # IntegerModel(distinct_numbers=distinct_frequencies, embedding_size=mapped_frequency_dim)
        if use_mean_to_map_nwf:
            self.mean_sequence = MeanOfList()
        else:
            self.map_sequence = RNNOfList(embedding_size=mapped_number_dim + mapped_frequency_dim,
                                          hidden_size=output_dim,
                                          num_layers=1, device=self.device)
        self.to(self.device)

    def forward(self, nwf_list, nf_name="unknown"):

        if len(nwf_list) > 0:
            mapped_frequencies = torch.cat([
                self.map_number([nwf['number'] for nwf in nwf_list]),
                self.map_frequency([nwf['frequency'] for nwf in nwf_list])], dim=1)

            if use_mean_to_map_nwf:
                return self.mean_sequence(mapped_frequencies)
            else:
                return self.map_sequence(mapped_frequencies)
        else:
            variable = torch.zeros(1, self.embedding_size,requires_grad=True).to(self.device)
            return variable



def configure_mappers(ploidy, extra_genotypes, num_samples, device, sample_dim=64, count_dim=64):
    """Return a tuple with two elements:
    mapper-dictionary: key is name of message type. value is function to map the message.
    all-modules: list of modules that implement mapping. """

    num_counts = ploidy + extra_genotypes

    map_CountInfo = MapCountInfo(mapped_count_dim=5, count_dim=count_dim, mapped_base_dim=2,
                                 mapped_genotype_index_dim=2, device=device)
    map_SampleInfo = MapSampleInfo(count_mapper=map_CountInfo, num_counts=num_counts, count_dim=count_dim,
                                   sample_dim=sample_dim, device=device)
    map_SbiRecords = MapBaseInformation(sample_mapper=map_SampleInfo, num_samples=num_samples, sample_dim=sample_dim,
                                        sequence_output_dim=count_dim, ploidy=ploidy, extra_genotypes=extra_genotypes,
                                        device=device)

    sbi_mappers = {"BaseInformation": map_SbiRecords,
                   "SampleInfo": map_SampleInfo,
                   "CountInfo": map_CountInfo
                   }
    return sbi_mappers, [map_SbiRecords, map_SampleInfo, map_CountInfo]
