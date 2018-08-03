from math import log2, log10, log

import torch
from torch.nn import Module

from org.campagnelab.dl.genotypetensors.structured.Models import Reduce, IntegerModel, map_Boolean, RNNOfList, \
    StructuredEmbedding, MeanOfList, LoadedTensor, LoadedBoolean


def store_indices_in_message(mapper, message, indices):
    mapper_id = id(mapper)
    if mapper_id not in message['indices']:
        message['indices'][mapper_id] = []
    message['indices'][mapper_id] += indices


def get_indices_in_message(mapper, message):
    return message['indices'][id(mapper)]


use_mean_to_map_nwf = False


class LoadedSequence(LoadedTensor):
    def __init__(self, sequence, base_to_index):
        # we cache a tensor of dimension 1 x seq_length, with one byte per base:

        super(LoadedSequence, self).__init__(
            torch.Tensor(list([base_to_index[b] for b in sequence])).type(dtype=torch.int8))


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
        self.bases = bases
        self.to(self.device)

    def forward(self, sequence_field):
        return self.map_sequence(
            self.map_bases(list([self.base_to_index[b] for b in sequence_field]))
        )

    def preload(self, sequence_field):
        return LoadedSequence(sequence_field, self.base_to_index)

    def loaded_forward(self, preloaded):
        """preloaded must have dimension batch x 1 x seq_length. We map each integer encoding of a base with map_bases to
        obtain dimension batch x encoding_dim x seq_length then map the sequence through the RNN. """
        return self.map_sequence.forward(
            self.map_bases.loaded_forward(preloaded))

    def simple_forward(self, tensor):
        return self.map_sequence.forward(
            self.map_bases.simple_forward(tensor))


class LoadedMapBaseInformation(LoadedTensor):
    def __init__(self, referenceBase, genomicSequenceContext, sequenceFrom, samples, base_to_index, num_samples,
                 mapper):
        super(LoadedMapBaseInformation, self).__init__(tensor=None, mapper=mapper)
        self.referenceBase = LoadedSequence(referenceBase, base_to_index=base_to_index)
        self.fromSequence = LoadedSequence(sequenceFrom, base_to_index=base_to_index)
        self.genomicSequenceContext = LoadedSequence(genomicSequenceContext, base_to_index=base_to_index)
        self.samples = [mapper.sample_mapper.preload(sample) for sample in samples]
        assert len(self.samples) == num_samples, "num_samples must match at this time. Sample padding not implemented."

        self.sample_from = LoadedSequence(sequenceFrom, base_to_index=base_to_index)

    def to(self, device, non_blocking=True):
        self.referenceBase = self.referenceBase.to(device, non_blocking=True)
        self.genomicSequenceContext = self.genomicSequenceContext.to(device, non_blocking=True)
        self.samples = [sample.to(device, non_blocking=True) for sample in self.samples]
        self.fromSequence = self.fromSequence.to(device=device, non_blocking=True)
        return self

    def tensor(self):
        """Return the tensor. """
        return self.mapper.loaded_forward(self)


class MapBaseInformation(StructuredEmbedding):
    def __init__(self, sample_mapper, sample_dim, num_samples, ploidy=2, extra_genotypes=2,
                 device=torch.device('cpu')):
        super().__init__(sample_dim, device)
        self.sample_mapper = sample_mapper

        # We reuse the same sequence mapper as the one used in MapCountInfo:
        self.map_sequence = sample_mapper.count_mapper.map_sequence
        sequence_output_dim = self.map_sequence.embedding_size
        self.reduce_samples = Reduce(
            [sequence_output_dim] + [sequence_output_dim] + [sequence_output_dim] + [sample_dim] * num_samples,
            encoding_output_dim=sample_dim, device=device)

        self.num_samples = num_samples
        self.ploidy = ploidy
        self.extra_genotypes = extra_genotypes

    def preload(self, input):
        return LoadedMapBaseInformation(input['referenceBase'], input['genomicSequenceContext'],
                                        input['samples'][0]['counts'][0]['fromSequence'],
                                        samples=[sample for sample in input['samples'][0:self.num_samples]],
                                        base_to_index=self.map_sequence.base_to_index,
                                        num_samples=self.num_samples,
                                        mapper=self)

    def forward(self, input):
        mapped_from = self.map_sequence(input['samples'][0]['counts'][0]['fromSequence'])
        return self.reduce_samples(
            [self.map_sequence(input['referenceBase'])] +
            [self.map_sequence(input['genomicSequenceContext'])] +
            # each sample has a unique from (same across all counts), which gets mapped and concatenated
            # with the sample mapped reduction:
            [mapped_from] +
            [self.sample_mapper(sample) for sample in input['samples'][0:self.num_samples]])

    def loaded_forward(self, preloaded):
        return self.reduce_samples(
            [self.map_sequence.loaded_forward(preloaded.referenceBase)] +
            [self.map_sequence.loaded_forward(preloaded.genomicSequenceContext)] +
            # each sample has a unique from (same across all counts), which gets mapped and concatenated
            # only once here:
            [self.map_sequence.loaded_forward(preloaded.fromSequence)] +
            # the sample mapper will not map fromSequence:
            [self.sample_mapper.loaded_forward(sample) for sample in preloaded.samples])


class LoadedSample(LoadedTensor):
    def __init__(self, loaded_counts, mapper):
        super(LoadedSample, self).__init__(tensor=None, mapper=mapper)
        self.counts = loaded_counts

    def to(self, device, non_blocking=True):
        self.counts = [count.to(device, non_blocking=True) for count in self.counts]
        return self


class MapSampleInfo(StructuredEmbedding):
    def __init__(self, count_mapper, count_dim, sample_dim, num_counts, device=None):
        super(MapSampleInfo, self).__init__(sample_dim,device=device)

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

    def preload(self, input):
        observed_counts = self.get_observed_counts(input)

        loaded_counts = [self.count_mapper.preload(count) for count in observed_counts[0:self.num_counts]]

        return LoadedSample(
            loaded_counts=loaded_counts,
            mapper=self)

    def loaded_forward(self, preloaded):
        return self.reduce_counts([self.count_mapper.loaded_forward(count) for count in
                                   preloaded.counts],
                                  pad_missing=True)

    def fold(self, fold, prefix, preloaded):
        reduced_counts=[self.count_mapper.fold(fold,prefix,count) for count in
                                   preloaded.counts]
        # emulate pad_missing:
        while len(reduced_counts)<len(self.reduce_counts.input_dims):
            zeros = torch.zeros(1, self.count_dim).to(device=self.device)
            reduced_counts.append(zeros)
        return fold.add(prefix+"_sample_reduce_count",*reduced_counts)

class LoadedZeros(LoadedTensor):
    def __init__(self, shape):
        super(LoadedZeros, self).__init__(torch.zeros(*shape, requires_grad=True))


class LoadedCount(LoadedTensor):
    def __init__(self, mapper, **args):
        self.mapper = mapper
        self.leaf_tensors = args

    def to(self, device, non_blocking=True):
        for key in self.leaf_tensors.keys():
            try:
                source = self.leaf_tensors[key]
                self.leaf_tensors[key] = source.to(device, non_blocking=non_blocking)
            except Exception as e:
                print("Error moving key {} to device {}: {}".format(key, device, e))
        return self

    def evaluate(self):
        self.mapper.loaded_forward()


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
                variable = variable.to(self.device)
                mapped += [variable]
        return self.reduce_count(mapped)

    def preload(self, c):
        mapped = {}
        for nf_name, mapper in self.nf_names_mappers:
            if nf_name in c.keys():
                mapped[nf_name] = mapper.preload(c[nf_name])
            else:
                mapped[nf_name] = LoadedZeros(shape=(1, mapper.embedding_size))

        return LoadedCount(mapper=self,
                           gobyGenotypeIndex=self.map_gobyGenotypeIndex.preload([c['gobyGenotypeIndex']]),
                           isIndel=self.map_boolean.preload(c['isIndel']),
                           matchesReference=self.map_boolean.preload(c['matchesReference']),
                           toSequence=self.map_sequence.preload(c['toSequence']),
                           genotypeCountForwardStrand=self.map_count.preload([c['genotypeCountForwardStrand']]),
                           genotypeCountReverseStrand=self.map_count.preload([c['genotypeCountReverseStrand']]),
                           **mapped
                           )

    def loaded_forward(self, preloaded):
        mapped_gobyGenotypeIndex = self.map_gobyGenotypeIndex.loaded_forward(
            preloaded.leaf_tensors['gobyGenotypeIndex'])
        # Do not map isCalled, it is a field that contains the truth and is used to calculate the label.

        mapped_isIndel = self.map_boolean.loaded_forward(preloaded.leaf_tensors['isIndel'])
        mapped_matchesReference = self.map_boolean.loaded_forward(preloaded.leaf_tensors['matchesReference'])

        # NB: fromSequence was mapped at the level of BaseInformation.
        mapped_to = self.map_sequence.loaded_forward(preloaded.leaf_tensors['toSequence'])
        mapped_genotypeCountForwardStrand = self.map_count.loaded_forward(
            preloaded.leaf_tensors['genotypeCountForwardStrand'])
        mapped_genotypeCountReverseStrand = self.map_count.loaded_forward(
            preloaded.leaf_tensors['genotypeCountReverseStrand'])

        mapped = [mapped_gobyGenotypeIndex,
                  mapped_isIndel,
                  mapped_matchesReference,
                  mapped_to,
                  mapped_genotypeCountForwardStrand,
                  mapped_genotypeCountReverseStrand]

        for nf_name, mapper in self.nf_names_mappers:
            mapped += [mapper.loaded_forward(preloaded.leaf_tensors[nf_name])]

        return self.reduce_count(mapped)

    def fold(self, fold, prefix, preloaded):
        prefix += "_count"
        mapped_gobyGenotypeIndex = fold.add(prefix + "_map_gobyGenotypeIndex",
                                            preloaded.leaf_tensors['gobyGenotypeIndex'].tensor())
        # Do not map isCalled, it is a field that contains the truth and is used to calculate the label.

        mapped_isIndel =  preloaded.leaf_tensors['isIndel'].tensor()
        mapped_matchesReference = preloaded.leaf_tensors['matchesReference'].tensor()

        # NB: fromSequence was mapped at the level of BaseInformation.
        mapped_to = fold.add(prefix + "_map_sequence", preloaded.leaf_tensors['toSequence'].tensor())

        mapped_genotypeCountForwardStrand = fold.add(prefix + "_map_count",
                                                     preloaded.leaf_tensors['genotypeCountForwardStrand'].tensor())
        mapped_genotypeCountReverseStrand = fold.add(prefix + "_map_count",
                                                     preloaded.leaf_tensors['genotypeCountReverseStrand'].tensor())
        mapped = [mapped_gobyGenotypeIndex,
                  mapped_isIndel,
                  mapped_matchesReference,
                  mapped_to,
                  mapped_genotypeCountForwardStrand,
                  mapped_genotypeCountReverseStrand]

        for nf_name, mapper in self.nf_names_mappers:
            mapped += [mapper.fold(fold, prefix+"_"+nf_name, preloaded.leaf_tensors[nf_name])]

        return fold.add(prefix + '_reduce_count', *mapped)


class FrequencyMapper(StructuredEmbedding):
    def __init__(self, device=None):
        super().__init__(3, device=device)
        self.LOG10 = log(10)
        self.LOG2 = log(2)
        self.epsilon = 1E-5

    def convert_list_of_floats(self, values, ignore_device=False):
        """This method accepts a list of floats."""
        x = torch.Tensor(values).view(-1, 1)
        if hasattr(self, 'epsilon'):
            x = x + self.epsilon
        if ignore_device:
            return x
        else:
            return x.to(self.device)

    def forward(self, x, ignore_device=False):
        """We use two floats to represent each number (the natural log of the number and the number divided by 10). The input must be a Float tensor of dimension:
        (num-elements in list, 1), or a list of float values.
        """
        if not torch.is_tensor(x):
            x = self.convert_list_of_floats(x, ignore_device)

        x = torch.cat([torch.log(x) / self.LOG10, torch.log(x) / self.LOG2, x / 10.0], dim=1)

        return x

    def preload(self, x):
        return LoadedTensor(self.forward(x, ignore_device=True))

    def simple_forward(self, x):

        x = x.view(-1, 1)
        if hasattr(self, 'epsilon'):
            x = x + self.epsilon

        x = torch.cat([torch.log(x) / self.LOG10, torch.log(x) / self.LOG2, x / 10.0], dim=1)

        return x

class LoadedNumberWithFrequency(LoadedTensor):
    def __init__(self, numbers, frequencies, mapper):
        self.mapper = mapper
        self.numbers = numbers
        self.frequencies = frequencies

    def to(self, device, non_blocking=True):
        self.numbers = self.numbers.to(device, non_blocking=non_blocking)
        self.frequencies = self.frequencies.to(device, non_blocking=non_blocking)
        return self

    def tensor(self):
        assert False, "tensor() called on LoadedNumberWithFrequency"
        # return torch.cat([self.numbers.tensor(), self.frequencies.tensor()],dim=0)


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
            variable = torch.zeros(1, self.embedding_size, requires_grad=True).to(self.device)
            return variable

    def preload(self, nwf_list):
        if len(nwf_list) > 0:
            return LoadedNumberWithFrequency(numbers=self.map_number.preload([nwf['number'] for nwf in nwf_list]),
                                             frequencies=self.map_frequency.preload(
                                                 [nwf['frequency'] for nwf in nwf_list]),
                                             mapper=self)
        else:
            return LoadedZeros(shape=(1, self.embedding_size))

    def loaded_forward(self, preloaded):
        if hasattr(preloaded, 'numbers'):
            mapped_frequencies = torch.cat([
                self.map_number.loaded_forward(preloaded.numbers),
                self.map_frequency.loaded_forward(preloaded.frequencies)], dim=1)

            if use_mean_to_map_nwf:
                return self.mean_sequence(mapped_frequencies)
            else:
                return self.map_sequence(mapped_frequencies)
        else:
            return preloaded.tensor()

    def fold(self, fold, prefix, preloaded):
        #prefix += "_nwf"
        if hasattr(preloaded, 'numbers'):

           return fold.add(prefix + "_map_nwl", preloaded.numbers.tensor(),preloaded.numbers.tensor().type(torch.float32))

        else:
            return preloaded.tensor()


class LoadedList(LoadedTensor):
    def __init__(self, loaded_tensors):
        self.loaded_tensors = loaded_tensors

    def to(self, device, non_blocking=True):
        for tensor in self.loaded_tensors:
            tensor.to(device, non_blocking=non_blocking)
        return self

    def tensor(self):
        return torch.cat([t.tensor() for t in self.loaded_tensors], dim=1)

    def __getitem__(self, item):
        return self.loaded_tensors[item]


class BatchOfRecords(Module):
    """Takes a list of structure instances and produce a tensor of size batch x embedding dim of each instance."""

    def __init__(self, sbi_mapper, device):
        """

        :param mappers: a dictionary, mapping message type to its mapper.
        """
        super().__init__()

        self.sbi_mapper = sbi_mapper
        self.to(device)

    def forward(self, preloaded_instance_list):
        mapped = [self.sbi_mapper.loaded_forward(instance) for instance in preloaded_instance_list]
        return torch.cat(mapped, dim=0)

    def preload(self, records):
        preloaded = LoadedList([self.sbi_mapper.preload(instance) for instance in records])
        return preloaded

    def loaded_forward(self, preloaded_records):
        return preloaded_records.tensor()


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
                                        ploidy=ploidy, extra_genotypes=extra_genotypes,
                                        device=device)

    sbi_mappers = {"BaseInformation": map_SbiRecords,
                   "SampleInfo": map_SampleInfo,
                   "CountInfo": map_CountInfo
                   }
    return sbi_mappers, [map_SbiRecords, map_SampleInfo, map_CountInfo]
