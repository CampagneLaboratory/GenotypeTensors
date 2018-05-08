import torch


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
        self.batched_results={}

    def collect_inputs(self, mapper, example,phase=0):
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
        input_for_mapper=mapper.collect_inputs(example,phase=phase,batcher=self)
        if isinstance(input_for_mapper,dict):
            # append to lists already obtained for the same mapper:
            for key in input_for_mapper.keys():
                self.initialize_mapper_variables(key)

                self.mapper_inputs[key]+=[input_for_mapper[key]]
        elif isinstance(input_for_mapper,list):
            self.mapper_inputs[id_mapper] += input_for_mapper
        else:
            self.mapper_inputs[id_mapper] += [input_for_mapper]

        current_example_index=self.example_counter[id_mapper]
        self.example_counter[id_mapper] += 1
        return current_example_index

    def initialize_mapper_variables(self, id_mapper):
        if id_mapper not in self.mapper_inputs.keys():
            self.mapper_inputs[id_mapper] = []
            self.example_counter[id_mapper] = 0

    def get_batched_input(self, mapper):
        """Get the batched input for a mapper."""
        id_mapper = id(mapper)
        mapper_inputs = self.mapper_inputs[id_mapper]
        if isinstance(mapper_inputs[0],str):
            if 'pending' in mapper_inputs :
                return None
        return torch.stack(mapper_inputs, dim=0)

    def forward_batch(self, mapper,phase=0):
        id_mapper = id(mapper)
        self.batched_results[id_mapper]= mapper.forward_batch(batcher=self,phase=phase)
        return self.batched_results[id_mapper]

    def get_forward_for_example(self,mapper, example_index ):
        """
        Return the slice of the forward result corresponding to a particular example.
        :param mapper: mapper used to do the forward pass.
        :param example_index: index of the example (returned by collect input)
        :return:
        """
        batch = self.get_batched_result(mapper)
        return batch[example_index:example_index + 1].squeeze(0)

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