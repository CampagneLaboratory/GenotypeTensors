import torch
from torch.autograd import Variable

from org.campagnelab.dl.genotypetensors.structured.Models import Reduce, IntegerModel, map_Boolean, RNNOfList


def configure_mappers(ploidy, extra_genotypes, num_samples):
    """Return a tuple with two elements:
    mapper-dictionary: key is name of message type. value is function to map the message.
    all-modules: list of modules that implement mapping. """
    all_modules=[]
    sample_dim = 64
    count_dim = 64
    num_counts = ploidy + extra_genotypes
    reduce_samples = Reduce([sample_dim] * num_samples, encoding_output_dim=sample_dim)
    reduce_counts = Reduce([count_dim] * num_counts, encoding_output_dim=count_dim)
    all_modules+=[reduce_samples,reduce_counts]

    mapped_genotype_index_dim = 2
    mapped_count_dim = 5
    mapped_isCalled_dim = 2
    mapped_base_dim = 2
    map_sequence = RNNOfList(embedding_size=mapped_base_dim, hidden_size=64, num_layers=1)
    map_gobyGenotypeIndex = IntegerModel(distinct_numbers=10, embedding_size=mapped_genotype_index_dim)
    all_modules += [map_sequence, map_gobyGenotypeIndex]

    bases = ['A', 'C', 'T', 'G', '-']
    max_base_index = 0
    for base in bases:
        max_base_index = max(ord(base[0]), max_base_index)
    map_bases = IntegerModel(distinct_numbers=(max_base_index + 1), embedding_size=mapped_base_dim)
    map_count = IntegerModel(distinct_numbers=100000, embedding_size=mapped_count_dim)
    map_isCalled = map_Boolean()
    map_isIndel = map_Boolean()
    map_matchesReference = map_Boolean()
    all_modules += [map_bases,map_count,map_isCalled,map_isIndel,map_matchesReference,]

    count_mappers = [map_gobyGenotypeIndex,
                     map_isCalled,
                     map_isIndel,
                     map_matchesReference,
                     map_sequence,
                     map_sequence,
                     map_count,
                     map_count]


    # below, [2+2+2] is for the booleans mapped with a function:
    reduce_count = Reduce([mapper.embedding_size for mapper in count_mappers], encoding_output_dim=count_dim)
    all_modules += [reduce_count]

    def map_BaseInformation(m):
        return reduce_samples([map_SampleInfo(sample) for sample in m['samples']])

    def map_SampleInfo(s):
        return reduce_counts([map_CountInfo(count) for count in s['counts']])

    def map_CountInfo(c):
        mapped_gobyGenotypeIndex = map_gobyGenotypeIndex([c['gobyGenotypeIndex']])
        mapped_isCalled = map_isCalled(c['isCalled'])
        mapped_isIndel = map_isIndel(c['isIndel'])
        mapped_matchesReference = map_matchesReference(c['matchesReference'])

        mapped_from = map_sequence(map_bases(list([ord(b) for b in c['fromSequence']])))
        mapped_to = map_sequence(map_bases(list([ord(b) for b in c['toSequence']])))
        mapped_genotypeCountForwardStrand = map_count([c['genotypeCountForwardStrand']])
        mapped_genotypeCountReverseStrand = map_count([c['genotypeCountReverseStrand']])

        return reduce_count([mapped_gobyGenotypeIndex,
                             mapped_isCalled,
                             mapped_isIndel,
                             mapped_matchesReference,
                             mapped_from,
                             mapped_to,
                             mapped_genotypeCountForwardStrand,
                             mapped_genotypeCountReverseStrand
                             ])

    sbi_mappers = {"BaseInformation": map_BaseInformation,
                   "SampleInfo": map_SampleInfo,
                   "CountInfo": map_CountInfo
                   }
    return sbi_mappers, all_modules
