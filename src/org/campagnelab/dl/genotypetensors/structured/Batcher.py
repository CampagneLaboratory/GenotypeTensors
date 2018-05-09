import torch
from torch.autograd import Variable


class Batcher:
    def __init__(self):
        self.new_batch()

    def new_batch(self):
        # a dictionary from mapper id to examples seen so far in a batch:
        self.example_counter = {}
        self.all_mappers = set()
        # a dictionary from mapper id to list of inputs for this mapper:
        self.mapper_inputs = {}
        # a dictionary from mapper id to results of forward pass on a batch:
        self.batched_results = {}

    def store_batched_result(self, mapper, batched_result):
        self.batched_results[id(mapper)] = batched_result

    def store_inputs(self, mapper, inputs):
        """
        Store inputs in the batcher.
        :param mapper: The mapper that produced these inputs.
        :param inputs: A torch Variable calculated by a mapper.
        :return: indices of the stored inputs in the batch under construction. These indices can be used to retrieve the
        slice of batched forward results corresponding to these inputs.
        """
        assert isinstance(inputs,Variable),"Inputs must be of type"
        id_mapper = id(mapper)
        self.initialize_mapper_variables(id_mapper)
        self.mapper_inputs[id_mapper] += [inputs]
        input_num_values = 1
        current_index = self.example_counter[id_mapper]
        self.example_counter[id_mapper] += input_num_values
        return list(range(current_index, current_index + input_num_values))

    def collect_inputs(self, mapper, example, phase=0):
        """
        Use the mapper on an example to obtain inputs for a batch.
        :param mapper:
        :param example:
        :return: a dictionary where each key is the name of a batch of inputs, and the value the list of inputs in that batch.
        """
        id_mapper = id(mapper)
        self.initialize_mapper_variables(id_mapper)

        # keep track of all mappers used:
        self.all_mappers.update([mapper])
        input_indices_in_batch = mapper.collect_inputs(example, phase=phase, batcher=self)
        return input_indices_in_batch

    def initialize_mapper_variables(self, id_mapper):
        if id_mapper not in self.mapper_inputs.keys():
            self.mapper_inputs[id_mapper] = []
            self.example_counter[id_mapper] = 0

    def get_batched_input(self, mapper):
        """Get the batched input for a mapper."""
        id_mapper = id(mapper)
        mapper_inputs = self.mapper_inputs[id_mapper]
        if mapper_inputs is None or isinstance(mapper_inputs, list) and len(mapper_inputs) == 0:
            return None

        if len(mapper_inputs) == 1:
            return mapper_inputs[0]
        #else:
        #    if isinstance(mapper_inputs, list) and all([isinstance(x,list) for x in mapper_inputs]):
        #        return mapper_inputs
        return torch.stack(mapper_inputs, dim=0)

    def forward_batch(self, mapper, phase=0):
        id_mapper = id(mapper)
        self.batched_results[id_mapper] = mapper.forward_batch(batcher=self, phase=phase)
        return self.batched_results[id_mapper]

    def get_forward_for_example(self, mapper, example_indices=None, message=None):
        """
        Return the slice of the forward result corresponding to a particular example.
        :param mapper: mapper used to do the forward pass.
        :param example_indices: index of the example (returned by collect input)
        :return:
        """
        if message is not None and example_indices is None:
            example_indices = message['indices'][id(mapper)]
        assert example_indices is not None, "example_indices is required when message is None."
        batch = self.get_batched_result(mapper)
        if isinstance(batch, dict):
            batch = batch[id(mapper)]
        return batch[example_indices].squeeze(0)

    def get_batched_result(self, mapper):
        """
        Return a previously calculated batch result.
        :param mapper: The mapper that calculated the result when its forward_batch method was called.
        :return: The batch.
        """
        id_mapper = id(mapper)
        assert id_mapper in self.batched_results, "mapped result not yet calculated for mapper id {}".format(id(mapper))
        batch = self.batched_results[id_mapper]
        return batch
