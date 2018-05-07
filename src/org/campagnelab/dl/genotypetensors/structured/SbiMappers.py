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

    def forward(self, sequence_field,tensor_cache, cuda=None):
        return self.map_sequence(self.map_bases(list([ord(b) for b in sequence_field]), tensor_cache=tensor_cache,cuda=cuda), cuda)


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

    def forward(self, input, tensor_cache,cuda=None):
        if cuda is None:
            cuda = next(self.parameters()).data.is_cuda

        return self.reduce_samples([self.map_sequence(input['referenceBase'],tensor_cache=tensor_cache,cuda=cuda)] +
                                   [self.map_sequence(input['genomicSequenceContext'],tensor_cache=tensor_cache, cuda=cuda)] +
                                   [self.sample_mapper(sample, tensor_cache=tensor_cache,cuda=cuda) for sample in
                                    input['samples'][0:self.num_samples]], cuda)


class MapSampleInfo(Module):
    def __init__(self, count_mapper, count_dim, sample_dim, num_counts):
        super().__init__()
        self.count_mapper = count_mapper
        self.num_counts = num_counts
        self.reduce_counts = Reduce([count_dim] * num_counts, encoding_output_dim=sample_dim)

    def forward(self, input,tensor_cache, cuda=None):
        observed_counts = [count for count in input['counts'] if
                           (count['genotypeCountForwardStrand'] + count['genotypeCountReverseStrand']) > 0]
        return self.reduce_counts([self.count_mapper(count,tensor_cache=tensor_cache, cuda=cuda) for count in
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

        count_mappers = [self.map_gobyGenotypeIndex,
                         self.map_boolean,  # isIndel
                         self.map_boolean,  # matchesReference
                         self.map_sequence,
                         self.map_sequence,
                         self.map_count,
                         self.map_count]

        # below, [2+2+2] is for the booleans mapped with a function:
        self.reduce_count = Reduce([mapper.embedding_size for mapper in count_mappers], encoding_output_dim=count_dim)

    def forward(self, c, tensor_cache, cuda=None):
        mapped_gobyGenotypeIndex = self.map_gobyGenotypeIndex([c['gobyGenotypeIndex']], tensor_cache=tensor_cache,cuda=cuda)
        # Do not map isCalled, it is a field that contains the truth and is used to calculate the label.

        mapped_isIndel = self.map_boolean(c['isIndel'],tensor_cache=tensor_cache,cuda=cuda)
        mapped_matchesReference = self.map_boolean(c['matchesReference'], tensor_cache=tensor_cache,cuda=cuda)

        mapped_from = self.map_sequence(c['fromSequence'],tensor_cache=tensor_cache,cuda=cuda)
        mapped_to = self.map_sequence(c['toSequence'],tensor_cache=tensor_cache,cuda=cuda)
        mapped_genotypeCountForwardStrand = self.map_count([c['genotypeCountForwardStrand']], tensor_cache=tensor_cache,cuda=cuda)
        mapped_genotypeCountReverseStrand = self.map_count([c['genotypeCountReverseStrand']],tensor_cache=tensor_cache,cuda=cuda)

        return self.reduce_count([mapped_gobyGenotypeIndex,
                                  mapped_isIndel,
                                  mapped_matchesReference,
                                  mapped_from,
                                  mapped_to,
                                  mapped_genotypeCountForwardStrand,
                                  mapped_genotypeCountReverseStrand
                                  ], cuda)


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
    return sbi_mappers, [map_SbiRecords,map_SampleInfo,map_CountInfo]
