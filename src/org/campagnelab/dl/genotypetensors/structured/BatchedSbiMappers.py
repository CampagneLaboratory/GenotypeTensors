from math import log2, log10, log

import torch
from torch.autograd import Variable
from torch.nn import Module, Embedding, LSTM, Linear

from org.campagnelab.dl.genotypetensors.autoencoder.struct_genotyping_supervised_trainer import sbi_json_string


def store_indices_in_message(mapper, message, indices):
    mapper_id = id(mapper)
    if mapper_id not in message['indices']:
        message['indices'][mapper_id] = []
    message['indices'][mapper_id] += indices


def get_indices_in_message(mapper, message):
    return message['indices'][id(mapper)]


class BatchedStructuredEmbedding(Module):
    def __init__(self, embedding_size, use_cuda=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        self.start_offsets = {}

    def cuda(self, device=0):
        self.use_cuda = True

    def define_long_variable(self, values, cuda=None):
        if isinstance(values, list):
            variable = Variable(torch.LongTensor(values), requires_grad=False)
        else:
            variable = Variable(torch.LongTensor([values]), requires_grad=False)

        if self.is_cuda(cuda):
            variable = variable.cuda(async=True)
        return variable

    def define_boolean(self, predicate):
        var = Variable(torch.FloatTensor([[1, 0]])) if predicate else Variable(torch.FloatTensor([[0, 1]]))
        if self.use_cuda:
            var = var.cuda()
        return var

    def is_cuda(self, cuda=None):
        """ Return True iff this model is on cuda. """
        return self.use_cuda

    def new_batch(self):
        pass

    def create_empty(self):
        variable = Variable(torch.zeros(1, self.embedding_size))
        if self.use_cuda:
            variable = variable.cuda()
        return variable

    def collect_tensors(self, elements, field_prefix, tensors, index_map={}):
        """

        :param elements:
        :param field_prefix:
        :param tensors:
        :return:
        """
        """Collect input tensors and store them in the list instances, keyed by the name of the field(s)
        For instance, counts.isIndel stores a list of Variables. counts.toSequence.bases stores Variables with long
        
        """
        assert isinstance(elements, list), "elements must be of type list"
        assert isinstance(field_prefix, str), "field_prefix must be of type str"
        assert isinstance(tensors, dict), "tensors must be of type dict"
        assert isinstance(index_map, dict), "index_map must be of type dict"

    def forward_batch(self, elements, field_prefix, tensors, index_maps={}):
        assert isinstance(index_maps, list), "index_maps must be a list of dict"
        assert len(index_maps) == 0 or isinstance(index_maps[0], dict), "index_maps must be a list of dict"

    def get_start_offset(self, list_name):
        if list_name not in self.start_offsets:
            self.start_offsets[list_name] = 0
            return 0
        else:
            index = self.start_offsets[list_name]
            return index

    def set_start_offset(self, list_name, index):
        self.start_offsets[list_name] = index

    def new_batch(self):
        self.start_offsets = {}

    def create_tensor_holder(self, tensors, field_prefix):
        if field_prefix not in tensors.keys():
            tensors[field_prefix] = []


class BooleanMapper(BatchedStructuredEmbedding):
    def __init__(self, ):
        super().__init__(2)

    def forward(self, tensors):
        return tensors

    def forward_batch(self, elements, field_prefix, tensors, index_maps=[]):
        super().forward_batch(elements, field_prefix, tensors, index_maps)

        self.create_tensor_holder(tensors, field_prefix)
        tensor_list = tensors[field_prefix]
        result = torch.cat(tensor_list, dim=0)
        tensors[field_prefix] = result
        return result

    def collect_tensors(self, elements, field_prefix, tensors, index_maps={}):
        assert isinstance(index_maps, dict), "index_maps must be of type dict"
        self.create_tensor_holder(tensors, field_prefix)

        if field_prefix not in index_maps.keys():
            index_maps[field_prefix] = []
        if field_prefix not in tensors:
            tensors[field_prefix] = []
        start_index = self.get_start_offset(field_prefix)
        if isinstance(elements, list):

            end = start_index + len(elements)
            index_maps[field_prefix] += range(start_index, end)

            tensor = torch.cat(
                [torch.FloatTensor([[1, 0]]) if value else torch.FloatTensor([[1, 0]]) for value in list])
            tensor = Variable(tensor, requires_grad=True)
            tensors[field_prefix].append(tensor)
        else:
            index_maps[field_prefix].append(start_index)
            end = start_index + 1
            tensor = torch.FloatTensor([[1, 0]]) if elements else torch.FloatTensor([[1, 0]])
            tensors[field_prefix].append(Variable(tensor, requires_grad=True))
        self.set_start_offset(field_prefix, end)


class IntegerMapper(BatchedStructuredEmbedding):
    def __init__(self, distinct_numbers, embedding_size):
        super().__init__(embedding_size)
        self.distinct_numbers = distinct_numbers
        self.embedding = Embedding(distinct_numbers, embedding_size)
        self.embedding.requires_grad = True

    def forward(self, values):
        """Accepts a list of integer values and produces a batch of batch x embedded-value. """

        assert isinstance(values, list), "values must be a list of integers."
        for value in values:
            assert value < self.distinct_numbers, "A value is larger than the embedding input allow: " + str(value)

        # cache the embedded values:
        cached_values = []
        for value in values:
            self.embedding(self.define_long_variable([value]))
        values = torch.cat(cached_values, dim=0)
        return values

    def collect_tensors(self, bases, field_name, tensors, index_maps={}):
        if isinstance(bases, int):
            bases = [bases]
        assert isinstance(bases, list) and isinstance(bases[0], int), "bases must contain a list of int."
        assert isinstance(index_maps, dict), "index_map must be of type dict"
        self.create_tensor_holder(tensors, field_name)
        # do not modify prefix, since we have no delegage fields.
        index = self.get_start_offset(field_name)

        if field_name not in index_maps:
            index_maps[field_name] = []
        end = index + len(bases)
        index_maps[field_name] += list(range(index, end))

        if field_name not in tensors:
            tensors[field_name] = []
        tensors[field_name].append(self.define_long_variable(bases))
        self.set_start_offset(field_name, end)

    def forward_batch(self, elements, field_prefix, tensors, index_maps=[]):
        super().forward_batch(elements, field_prefix, tensors, index_maps)

        self.create_tensor_holder(tensors, field_prefix)

        tensor_list = tensors[field_prefix]
        result = self.embedding(torch.cat(tensor_list, dim=0))
        tensors[field_prefix] = result
        return result


class MeanOfList(Module):
    def __init__(self):
        super().__init__()

    def forward(self, list_of_embeddings, cuda=None):
        assert isinstance(list_of_embeddings, Variable) or isinstance(list_of_embeddings,
                                                                      torch.FloatTensor), "input must be a Variable or Tensor"
        num_elements = list_of_embeddings.size(0)
        return (torch.sum(list_of_embeddings, dim=0) / num_elements).view(1, -1)


class RNNOfList(BatchedStructuredEmbedding):
    def __init__(self, embedding_size, hidden_size, num_layers=1):
        super().__init__(hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = LSTM(embedding_size, hidden_size, num_layers, batch_first=True)

    def forward(self, list_of_embeddings):

        batch_size = list_of_embeddings.size(0)
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        hidden = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        memory = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        if self.is_cuda():
            hidden = hidden.cuda(async=True)
            memory = memory.cuda(async=True)
        states = (hidden, memory)
        inputs = list_of_embeddings.contiguous()  # .view(batch_size, 1, -1)
        out, states = self.lstm(inputs, states)
        # return the last output:
        all_outputs = out.squeeze()
        if (all_outputs.dim() > 2):
            # return the last output:
            return all_outputs[:, -1, :].contiguous()
        else:
            return all_outputs.contiguous()

    def collect_tensors(self, elements, field_prefix, tensors, index_maps={}):
        assert isinstance(index_maps, dict), "index_map must be of type dict"
        self.create_tensor_holder(tensors, field_prefix)
        if field_prefix not in tensors:
            tensors[field_prefix] = []
        tensors[field_prefix].append(elements)

    def forward_batch(self, elements, field_prefix, tensors, index_maps=[]):
        self.create_tensor_holder(tensors, field_prefix)

        result = self.lstm(elements)
        tensors[field_prefix] = result
        return result


class Reduce(BatchedStructuredEmbedding):
    """ Reduce a list of embedded fields or messages using a feed forward. Requires list to be a constant size """

    def __init__(self, input_dims, encoding_output_dim):
        """

        :param input_dims: dimensions of the elements to reduce (list of size the number of elements, integer dimension).
        :param encoding_output_dim: dimension of the reduce output.
        """
        super().__init__(embedding_size=encoding_output_dim)
        self.encoding_dim = encoding_output_dim
        self.input_dims = input_dims
        sum_input_dims = 0
        for dim in input_dims:
            sum_input_dims += dim
        self.sum_input_dims = sum_input_dims
        self.linear1 = Linear(sum_input_dims, sum_input_dims * 2)
        self.linear2 = Linear(sum_input_dims * 2, encoding_output_dim)

    def forward(self, input_list, pad_missing=False):
        if len(input_list) != len(self.input_dims) and not pad_missing:
            raise ValueError("The input_list dimension does not match input_dims, and pad_missing is not specified. ")
        batch_size = input_list[0].size(0)
        if pad_missing:
            index = 0
            while len(input_list) < len(self.input_dims):
                # pad input list with minibatches of zeros:
                variable = Variable(torch.zeros(batch_size, self.input_dims[index]))
                if self.is_cuda():
                    variable = variable.cuda(async=True)
                input_list += [variable]
                index += 1

        if any([input.dim() != 2 for input in input_list]) or [input.size(1) for input in
                                                               input_list] != self.input_dims:
            print("STOP: input dims={} sizes={}".format([input.dim() for input in input_list],
                                                        [input.size() for input in input_list]))
        for index, input in enumerate(input_list):
            assert input.size(1) == self.input_dims[index], "input dimension must match declared for input {} ".format(
                index)
        x = torch.cat(input_list, dim=1)

        return self.forward_flat_inputs(x)

    def forward_flat_inputs(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        return x.view(-1, self.encoding_dim)


def store_indices_in_message(mapper, message, indices):
    mapper_id = id(mapper)
    if mapper_id not in message['indices']:
        message['indices'][mapper_id] = []
    message['indices'][mapper_id] += indices


def get_indices_in_message(mapper, message):
    return message['indices'][id(mapper)]


class MapSequence(BatchedStructuredEmbedding):
    def __init__(self, mapped_base_dim=2, hidden_size=64, num_layers=1, bases=('A', 'C', 'T', 'G', '-', 'N')):
        super().__init__(embedding_size=hidden_size)
        self.map_sequence = RNNOfList(embedding_size=mapped_base_dim, hidden_size=hidden_size,
                                      num_layers=num_layers)
        self.base_to_index = {}
        for base_index, base in enumerate(bases):
            self.base_to_index[base[0]] = base_index
        self.mapped_base_dim = mapped_base_dim
        self.map_bases = IntegerMapper(distinct_numbers=len(self.base_to_index), embedding_size=mapped_base_dim)

    def collect_tensors(self, sequence, field_name, tensors, index_maps={}):
        assert isinstance(index_maps, dict), "index_maps must be of type dict"
        self.create_tensor_holder(tensors, field_name)
        # two states possible: bases have been collected already, or they have not.
        field_name += ".sequence"
        index = self.get_start_offset(field_name)
        if field_name not in index_maps:
            index_maps[field_name] = []
        index_maps[field_name].append(index)
        self.map_bases.collect_tensors([self.base_to_index[b] for b in sequence], field_name + ".bases", tensors,
                                       index_maps)
        index = index + 1
        self.set_start_offset(field_name, index)

    def forward(self, sequence_field):
        return self.map_sequence(self.collect_tensors(sequence_field))

    def forward_batch(self, elements, field_prefix, tensors, index_maps=[]):
        # this method ignores index_maps and retrieves the map from the element passed in.
        super().forward_batch(elements, field_prefix, tensors, index_maps)
        field_prefix += ".sequence"
        self.create_tensor_holder(tensors, field_prefix)

        """Do a forward batch on bases. This will calculate embeddings on bases using batched Long tensors:"""
        batched_bases = self.map_bases.forward_batch(elements, field_prefix + ".bases", tensors)
        # print(batched_bases)
        sequence_tensors = []
        max_length = 0
        for index_map in index_maps:
            # we get the indices of the base embeddings (stored with .ints prefix)
            slice_indices = index_map[field_prefix + ".bases"]
            individual_sequence = batched_bases[slice_indices].view(1, -1, self.mapped_base_dim)
            max_length = max(max_length, individual_sequence.size(1))
            sequence_tensors.append(individual_sequence)

        if len(sequence_tensors) == 0:
            tensors[field_prefix] = self.create_empty().view(1, 1, self.embedding_size)
            return tensors[field_prefix]
        # Pad all sequences to max length:
        padded_tensors = []
        for sequence_tensor in sequence_tensors:
            length = sequence_tensor.size(1)
            padded = torch.nn.functional.pad(input=sequence_tensor, pad=(0, 0, 0, max_length - length),
                                             mode='constant', value=0)
            padded_tensors.append(padded)
        # print("padded: "+str(padded_tensors))

        # do forward on padded batched sequences:
        batched_paddded = torch.cat(padded_tensors, dim=0)
        mapped_sequences = self.map_sequence(batched_paddded)

        tensors[field_prefix] = mapped_sequences
        print("sequence mapped to size {}".format(mapped_sequences.size()))
        return mapped_sequences


class MapBaseInformation(BatchedStructuredEmbedding):
    def __init__(self, sample_mapper, sample_dim, num_samples, sequence_output_dim=64, ploidy=2, extra_genotypes=2):
        super().__init__(sample_dim)
        self.sample_mapper = sample_mapper

        # We reuse the same sequence mapper as the one used in MapCountInfo:
        self.map_sequence = sample_mapper.count_mapper.map_sequence
        sum_of_dims=self.map_sequence.embedding_size*2+(sample_dim)*num_samples
        self.reduce_samples = Reduce([sum_of_dims],
            encoding_output_dim=sample_dim)

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

    def collect_tensors(self, elements, field_prefix, tensors, index_map={}):
        super().collect_tensors(elements, field_prefix, tensors, index_map)
        field_prefix += ".record"
        offset = self.get_start_offset(field_prefix + ".list")
        for record in elements:
            if 'indices' not in record.keys():
                record['indices'] = {}
            # prepare list that will keep indices to each record:
            record['indices'][field_prefix + ".list"] = []
        index = offset
        for record_index, record in enumerate(elements):
            assert record['type'] == "BaseInformation", "This mapper only accepts type==BaseInformation"
            index_map = record['indices']
            index = record_index + offset

            index_map[field_prefix + ".list"].append(index)
            self.map_sequence.collect_tensors(record['referenceBase'], field_prefix + ".referenceBase", tensors,
                                              record['indices'])
            self.map_sequence.collect_tensors(record['genomicSequenceContext'],
                                              field_prefix + ".genomicSequenceContext", tensors, record['indices'])
            self.sample_mapper.collect_tensors(record['samples'], field_prefix + ".samples", tensors, None)
            index += 1
        self.set_start_offset(field_prefix + ".list", index)
        return tensors

    def forward_batch(self, elements, field_prefix, tensors, index_maps={}):
        # super().forward_batch(elements, field_prefix, tensors, index_maps)
        field_prefix += ".record"
        # Process all counts together:

        all_samples = []
        for record in elements:
            for sample in record['samples'][0:self.num_samples]:
                all_samples.append(sample)

        batched_mapped_samples = self.sample_mapper.forward_batch(elements=all_samples,
                                                                  field_prefix=field_prefix + ".samples",
                                                                  tensors=tensors,
                                                                  index_maps=[sample['indices'] for sample in
                                                                              all_samples])

        batched_reference_bases = self.map_sequence.forward_batch(elements=elements,
                                                                   field_prefix=field_prefix + ".referenceBase",
                                                                   tensors=tensors,
                                                                   index_maps=[record['indices'] for record in
                                                                              elements])

        batched_genomic_contexts = self.map_sequence.forward_batch(elements=elements,
                                                                   field_prefix=field_prefix + ".genomicSequenceContext",
                                                                   tensors=tensors,
                                                                   index_maps=[record['indices'] for record in
                                                                              elements])
        # Go through every record to assemble the slices:
        mapped_records = []
        for record in elements:
            mapped_samples = []
            for sample in record['samples'][0:self.num_samples]:

                sample_indices_for_record = sample['indices'][field_prefix + ".samples.sample.list"]
                mapped_samples.append(batched_mapped_samples[sample_indices_for_record])
            index_map = record['indices']
            sample_indices_for_record=index_map[field_prefix + ".list"]
            mapped_samples.append(batched_reference_bases[sample_indices_for_record])
            mapped_samples.append(batched_genomic_contexts[sample_indices_for_record])
            mapped_records.append(torch.cat(mapped_samples, dim=1))
        batched_sample_counts_inputs = torch.cat(mapped_records, dim=0)

        # reduce all counts in one shot:
        result = self.reduce_samples.forward_flat_inputs(batched_sample_counts_inputs)
        tensors["mapped$" + field_prefix] = result
        return result


class MapSampleInfo(BatchedStructuredEmbedding):

    def __init__(self, count_mapper, count_dim, sample_dim, num_counts):
        super().__init__(sample_dim)
        assert isinstance(count_mapper, MapCountInfo)
        self.count_mapper = count_mapper
        self.num_counts = num_counts
        self.count_dim = count_dim
        self.reduce_counts = Reduce([count_dim] * num_counts, encoding_output_dim=sample_dim)

    def collect_tensors(self, elements, field_prefix, tensors, index_map={}):
        #        super().collect_tensors(elements, field_prefix, tensors, index_map)
        field_prefix += ".sample"
        offset = self.get_start_offset(field_prefix + ".list")
        for sample in elements:
            if 'indices' not in sample.keys():
                sample['indices'] = {}
                # prepare list that will keep indices to each sample:
                sample['indices'][field_prefix + ".list"] = []
        index = offset
        for sample_index, sample in enumerate(elements):
            assert sample['type'] == "SampleInfo", "This mapper only accepts type==SampleInfo"

            index_map = sample['indices']
            index = sample_index + offset

            index_map[field_prefix + ".list"].append(index)
            all_counts = []
            for count_list in (sample['counts'] for sample in elements):
                for count in count_list:
                    print(count)
                    all_counts.append(count)
            self.count_mapper.collect_tensors(sample['counts'], field_prefix + ".counts", tensors, None)
            index += 1
        self.set_start_offset(field_prefix + ".list", index)
        return tensors

    def forward_batch(self, elements, field_prefix, tensors, index_maps={}):
        # super().forward_batch(elements, field_prefix, tensors, index_maps)
        field_prefix += ".sample"
        # Process all counts together:

        all_counts = []
        for count_list in (self.get_observed_counts(sample) for sample in elements):
            for count in count_list[0:self.num_counts]:
                print(count)
                all_counts.append(count)
        batched_mapped_counts = self.count_mapper.forward_batch(elements=all_counts,
                                                                field_prefix=field_prefix + ".counts", tensors=tensors,
                                                                index_maps=[count['indices'] for count in all_counts])

        # Go through every sample to assemble the slices:
        mapped_sample_counts = []
        for sample in elements:
            index_map = sample['indices']
            count_indices_for_sample = index_map[field_prefix + ".list"]
            mapped_sample_counts.append(batched_mapped_counts[count_indices_for_sample])

        batched_sample_counts_inputs = torch.cat(mapped_sample_counts, dim=1)

        # reduce all counts in one shot:
        result = self.reduce_counts.forward_flat_inputs(batched_sample_counts_inputs)
        tensors["mapped$" + field_prefix] = result
        return result

    def forward(self, input, tensor_cache, cuda=None):
        observed_counts = self.get_observed_counts(input)
        return self.reduce_counts([self.count_mapper(count) for count in
                                   observed_counts[0:self.num_counts]],
                                  pad_missing=True, cuda=cuda)

    def get_observed_counts(self, input):
        return [count for count in input['counts'] if
                (count['genotypeCountForwardStrand'] + count['genotypeCountReverseStrand']) > 0]


class MapCountInfo(BatchedStructuredEmbedding):
    def __init__(self, mapped_count_dim=5, count_dim=64, mapped_base_dim=2, mapped_genotype_index_dim=4,
                 use_cuda=False):
        super().__init__(count_dim, use_cuda=use_cuda)

        self.frequency_list_mapper_base_qual = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_num_var = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_mapping_qual = MapNumberWithFrequencyList(distinct_numbers=100)
        self.frequency_list_mapper_distance_to = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_aligned_lengths = MapNumberWithFrequencyList(distinct_numbers=1000)
        self.frequency_list_mapper_read_indices = MapNumberWithFrequencyList(distinct_numbers=1000)

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
        self.all_fields = []  # a list of (field_name, field_mapper)

        self.all_fields += self.nf_names_mappers

        self.map_sequence = MapSequence(hidden_size=count_dim,
                                        mapped_base_dim=mapped_base_dim)
        self.map_gobyGenotypeIndex = IntegerMapper(distinct_numbers=100, embedding_size=mapped_genotype_index_dim)

        self.map_count = IntegerMapper(distinct_numbers=100000, embedding_size=mapped_count_dim)
        self.map_boolean = BooleanMapper()

        self.all_fields += [('toSequence', self.map_sequence),
                            ('gobyGenotypeIndex', self.map_gobyGenotypeIndex),
                            ('isIndel', self.map_boolean),
                            ('matchesReference', self.map_boolean),
                            ('genotypeCountForwardStrand', self.map_count),
                            ('genotypeCountReverseStrand', self.map_count)]

        # below, [2+2+2] is for the booleans mapped with a function:
        sum_of_dims = 0
        for _, mapper in self.all_fields:
            sum_of_dims += mapper.embedding_size
        print("sum_of_dims {}".format(sum_of_dims))
        self.reduce_count = Reduce([sum_of_dims],
                                   encoding_output_dim=count_dim)

    def new_batch(self):
        super().new_batch()
        for field_name, mapper in self.all_fields:
            mapper.new_batch()

    def collect_tensors(self, list_of_counts, field_prefix, tensors={}, index_maps={}):
        """Return a dict, where each value is a batched tensor, and the keys are the name of the field that this tensor was
        collected from. The index of each field in the tensor is stored in list_of_counts,
        modifying each field of each count to store index instead of the field. index is the index of the field value inside
        the batched returned tensor."""

        field_prefix += ".count"

        for field_name, _ in self.all_fields:
            if field_name not in tensors:
                tensors[field_prefix + "." + field_name] = []
        offset = self.get_start_offset(field_prefix)
        for count_index, count in enumerate(list_of_counts):
            assert count['type'] == "CountInfo", "This mapper only accepts type==CountInfo"

            if 'indices' not in count.keys():
                count['indices'] = {}

            index = count_index + offset
            for field_name, mapper in self.all_fields:
                print("Collecting {}".format(field_name), flush=True)
                mapper.collect_tensors(count[field_name], field_prefix + "." + field_name, tensors, count['indices'])
            self.set_start_offset(field_prefix, index + 1)
        return tensors

    def forward_batch(self, elements, field_prefix, tensors, index_maps=[]):
        super().forward_batch(elements, field_prefix, tensors, index_maps)

        field_prefix += ".count"

        self.create_tensor_holder(tensors, field_prefix)
        for nf_name, nf_mapper in self.nf_names_mappers:
            all_nwfs = []
            for count in elements:
                for element in count[nf_name]:
                    all_nwfs.append(element)
            self.create_tensor_holder(tensors, field_prefix + "." + nf_name)
            tensors["mapped$" + field_prefix + "." + nf_name] = nf_mapper.forward_batch(all_nwfs,
                                                                                        field_prefix + "." + nf_name,
                                                                                        tensors,

                                                                                        index_maps=[count['indices'] for
                                                                                                    count
                                                                                                    in elements])

        for field_name, mapper in self.all_fields:
            print("Mapping " + field_name, flush=True)
            if not isinstance(mapper, MapNumberWithFrequencyList):
                tensors["mapped$" + field_prefix + "." + field_name] = mapper.forward_batch(elements=elements,
                                                                                            field_prefix=field_prefix + "." + field_name,
                                                                                            tensors=tensors,
                                                                                            index_maps=[count['indices']
                                                                                                        for count in
                                                                                                        elements])

        # remove inputs to the nf mappers
        # for nf_name, _ in self.nf_names_mappers:
        #    tensors[field_prefix+nf_name]=None
        tensors_for_individual_counts = []
        # slice batched tensors into the instances of counts:

        for count in elements:
            print("count -----")
            slice_index = count['indices'][field_prefix + '.isIndel'][
                0]  # used a field of cardinality 1 to retrieve this count index

            count_components = []
            for field_name, _ in self.all_fields:

                batched_mapped_tensor = tensors["mapped$" + field_prefix + "." + field_name]
                if slice_index < batched_mapped_tensor.size(0):
                    tensor = batched_mapped_tensor[slice_index]
                    if tensor is not None:

                        print("processing {} field of size {}".format(field_name, tensor.view(1, -1).size()),
                              flush=True)
                        count_components.append(tensor.view(1, -1))
                    else:
                        print("Could not find tensor for field " + field_name)
                else:
                    print("Not enough elements for field " + field_prefix + "." + field_name)
            tensors_for_individual_counts.append(torch.cat(count_components, dim=1))

        batched_count_tensors = torch.cat(tensors_for_individual_counts, dim=0)
        result = self.reduce_count.forward_flat_inputs(batched_count_tensors)
        tensors[field_prefix] = result
        return result


class FrequencyMapper(BatchedStructuredEmbedding):
    def __init__(self):
        super().__init__(3)
        self.LOG10 = log(10)
        self.LOG2 = log(2)
        self.epsilon = 1E-5

    def convert_list_of_floats(self, values):
        """This method accepts a list of floats."""
        x = torch.FloatTensor(values).view(-1, 1)
        x = x + self.epsilon
        return x

    def collect_tensors(self, elements, field_prefix, tensors, index_maps={}):
        assert isinstance(index_maps, dict), "index_maps must be of type dict"

        if field_prefix not in index_maps:
            index_maps[field_prefix] = []
        start = self.get_start_offset(field_prefix)
        end = start + len(elements)
        index_maps[field_prefix] += list(range(start, end))
        self.set_start_offset(field_prefix, end)
        return tensors[field_prefix].append(self.forward(elements))

    def forward_batch(self, elements, field_prefix, tensors, index_maps=[]):
        super().forward_batch(elements, field_prefix, tensors, index_maps)

        self.create_tensor_holder(tensors, field_prefix)
        result = torch.cat(tensors[field_prefix], dim=0)
        tensors[field_prefix] = result
        return result

    def forward(self, x):
        """We use two floats to represent each number (the natural log of the number and the number divided by 10). The input must be a Float tensor of dimension:
        (num-elements in list, 1), or a list of float values.
        """
        if not torch.is_tensor(x):
            x = self.convert_list_of_floats(x)

        x = torch.cat([torch.log(x) / self.LOG10, torch.log(x) / self.LOG2, x / 10.0], dim=1)
        variable = Variable(x, requires_grad=True)
        if self.use_cuda:
            variable = variable.cuda(async=True)
        return variable


class MapNumberWithFrequencyList(BatchedStructuredEmbedding):
    def __init__(self, distinct_numbers=-1, mapped_number_dim=4):
        mapped_frequency_dim = 3
        super().__init__(embedding_size=mapped_number_dim + mapped_frequency_dim)

        output_dim = mapped_number_dim + mapped_frequency_dim
        self.map_number = IntegerMapper(distinct_numbers=distinct_numbers, embedding_size=mapped_number_dim)
        self.map_frequency = FrequencyMapper()
        # stores offsets across a batch key is list name, value is offset to use when generating new indices.
        self.start_offsets = {}
        # self.map_frequency = Variable(torch.FloatTensor([[]]))
        # IntegerModel(distinct_numbers=distinct_frequencies, embedding_size=mapped_frequency_dim)

        self.mean_sequence = MeanOfList()

    def collect_tensors(self, list_of_nwf, field_prefix, tensors, index_maps={}):
        """Return a dict, where each value is a batched tensor, and the keys are the name of the field that this tensor was
                collected from. The index of each field in the tensor is stored in list_of_counts,
                modifying each field of each count to store index instead of the field. index is the index of the field value inside
                the batched returned tensor."""
        assert isinstance(index_maps, dict), "index_maps must be of type dict"

        field_prefix += ".nwf"
        start_offset = self.get_start_offset(field_prefix)

        number_field = field_prefix + ".number"
        frequency_field = field_prefix + ".frequency"
        if number_field not in tensors:
            tensors[number_field] = []
            tensors[frequency_field] = []

        for nwf_index, nwf in enumerate(list_of_nwf):
            assert nwf[
                       'type'] == "NumberWithFrequency", "This mapper only accepts type==NumberWithFrequency, got " + str(
                nwf)
            # nwf_index here starts at zero, but this list of nwf may be collected as part of other lists.
            # we add the start offset to account for nw previously mapped in the same batch. start offset is reset
            # when calling new_batch()
            if 'indices' not in nwf:
                nwf['indices'] = {}
            offset = nwf_index + start_offset
            nwf['index'] = offset
            self.map_number.collect_tensors([nwf['number']], field_prefix + '.number', tensors, index_maps)
            self.map_frequency.collect_tensors([nwf['frequency']], field_prefix + '.frequency', tensors, index_maps)

        return tensors

    def forward_batch(self, elements, field_prefix, tensors, index_maps=[]):
        super().forward_batch(elements, field_prefix, tensors, index_maps)

        # map the batched inputs:
        field_prefix += ".nwf"
        self.create_tensor_holder(tensors, field_prefix)

        mapped_frequency = self.map_frequency.forward_batch(elements=elements, field_prefix=field_prefix + '.frequency',
                                                            tensors=tensors,
                                                            index_maps=[nwf['indices'] for nwf in elements])

        mapped_number = self.map_number.forward_batch(elements=elements, field_prefix=field_prefix + '.number',
                                                      tensors=tensors, index_maps=[nwf['indices'] for nwf in elements])
        mapped_tensors = []
        # then go through each element and assign the slices:
        for index_map in index_maps:
            index_map["mean-input$" + field_prefix] = []
            # do not concat empty fields:
            if (field_prefix + '.frequency') in index_map.keys():
                frequency_indices = index_map[field_prefix + '.frequency']
                number_indices = index_map[field_prefix + '.number']
                for frequency_index, number_index in zip(iter(frequency_indices), iter(number_indices)):
                    index_map["mean-input$" + field_prefix].append(
                        torch.cat([mapped_number[frequency_index].view(1, -1),
                                   mapped_frequency[number_index].view(1, -1)],
                                  dim=1))
            else:
                print("Appending empty for " + field_prefix)
                index_map["mean-input$" + field_prefix].append(torch.cat(
                    [self.map_number.create_empty().view(1, -1),
                     self.map_frequency.create_empty().view(1, -1)], dim=1))
            mapped_tensors.append(self.mean_sequence(torch.stack(index_map["mean-input$" + field_prefix])))

        result = torch.stack(mapped_tensors)
        tensors[field_prefix] = result
        return result


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


def map_count(list_of_counts, count_dim=64):
    map_CountInfo = MapCountInfo(mapped_count_dim=5, count_dim=count_dim, mapped_base_dim=2,
                                 mapped_genotype_index_dim=2)
    tensors = map_CountInfo.collect_tensors(list_of_counts, 'counts', {})

    map_CountInfo.forward_batch(list_of_counts, "counts", tensors)


if __name__ == '__main__':
    import ujson

    record = ujson.loads(sbi_json_string)
    map_count(record['samples'][0]['counts'])
