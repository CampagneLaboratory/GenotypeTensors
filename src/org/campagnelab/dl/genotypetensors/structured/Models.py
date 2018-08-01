import threading

import torch
from torch.autograd import Variable
from torch.nn import Module, Embedding, LSTM, Linear, ModuleList


class StructuredEmbedding(Module):
    def __init__(self, embedding_size, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cpu')
            self.using_cuda = False
        self.embedding_size = embedding_size
        self.device = device
        if device == torch.device('cuda'):
            self.using_cuda = True

    def define_long_variable(self, values):
        variable = torch.Tensor(values).type(dtype=torch.long).to(self.device)
        return variable

    def is_cuda(self, ):
        """ Return True iff this model is on cuda. """
        if self.self.using_cuda is not None:
            return self.using_cuda
        else:
            cuda = next(self.parameters()).data.is_cuda
        return cuda

    def loaded_forward(self, preloaded):
        """Accepts a preloaded instance (subclass of LoadedTensor) and returns a mapped tensor. """
        return preloaded.tensor()

class LoadedTensor:
    def __init__(self,tensor,mapper=None):
        self.tensor_=tensor
        self.mapper = mapper

    def to(self, device, non_blocking=True):
        """Move the tensor to the device. """

        self.tensor_ = self.tensor_.to(device)
        return self

    def tensor(self):
        """Return the tensor. """
        return self.tensor_

    def forward(self):
        """Call forward on the mapper with this pre-loaded tensor."""
        return self.mapper.loaded_forward(self)


class LoadedBoolean(LoadedTensor):
    def __init__(self, tensor):
        super(LoadedBoolean, self).__init__(tensor)

class LoadedLongs(LoadedTensor):
    def __init__(self,tensor):
        super(LoadedLongs, self).__init__(tensor)

class map_Boolean(StructuredEmbedding):
    def __init__(self, device):
        super().__init__(2, device)
        self.true_value = torch.FloatTensor([[1, 0]])
        self.false_value = torch.FloatTensor([[0, 1]])
        self.to(self.device)

    def forward(self, predicate):
        assert isinstance(predicate, bool), "predicate must be a boolean"
        value = torch.FloatTensor([[1, 0]]) if predicate else torch.FloatTensor([[0, 1]])
        value = value.to(self.device)
        return value
        # return Variable(value.data,requires_grad=True)
        # return value

    def preload(self, predicate):
        return LoadedBoolean(tensor=torch.FloatTensor([[1, 0]]) if predicate else torch.FloatTensor([[0, 1]]))



class IntegerModel(StructuredEmbedding):
    def __init__(self, distinct_numbers, embedding_size, device):
        super().__init__(embedding_size, device)
        self.distinct_numbers = distinct_numbers
        self.embedding = Embedding(distinct_numbers, embedding_size)
        self.embedding.requires_grad = True
        self.to(device)
        self.embedding.to(device)

    def forward(self, values):
        """Accepts a list of integer values and produces a batch of batch x embedded-value. """

        assert isinstance(values, list), "values must be a list of integers."
        for value in values:
            assert value < self.distinct_numbers, "A value is larger than the embedding input allow: " + str(value)

        # cache the embedded values:
        cached_values = []
        for value in values:
            cached_values += [self.embedding(self.define_long_variable([value]))]

        values = torch.cat(cached_values, dim=0)
        return values

    def preload(self,values):
        assert isinstance(values, list), "values must be a list of integers."
        for value in values:
            assert value < self.distinct_numbers, "A value is larger than the embedding input allow: " + str(value)

        # cache the embedded values:
        cached_values = []
        return LoadedLongs(self.define_long_variable(values))


    def loaded_forward(self, preloaded):
        """Accepts a preloaded tensor of dimension batch x 1 x length and produces a batch of batch x embedded-value x length. """
        tensor = preloaded.tensor().type(dtype=torch.long)
        return self.embedding(tensor)


class MeanOfList(Module):
    def __init__(self):
        super().__init__()

    def forward(self, list_of_embeddings, ):
        assert isinstance(list_of_embeddings, torch.Tensor) or isinstance(list_of_embeddings,
                                                                          torch.Tensor), "input must be a Variable or Tensor"
        num_elements = list_of_embeddings.size(0)
        return (torch.sum(list_of_embeddings, dim=0) / num_elements).view(1, -1)


class RNNOfList(StructuredEmbedding):
    def __init__(self, embedding_size, hidden_size, num_layers=1, device=None):
        super().__init__(hidden_size, device)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.to(self.device)

    def forward(self, list_of_embeddings, ):

        batch_size = list_of_embeddings.size(0)
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        hidden = torch.zeros(num_layers, batch_size, hidden_size).to(self.device)
        memory = torch.zeros(num_layers, batch_size, hidden_size).to(self.device)
        states = (hidden, memory)
        inputs = list_of_embeddings.view(batch_size, 1, -1)
        out, states = self.lstm(inputs, states)
        # return the last output:
        all_outputs = out.squeeze()
        if (all_outputs.dim() > 1):
            # return the last output:
            return all_outputs[-1, :].view(1, -1)
        else:
            return all_outputs.view(1, -1)


class Reduce(StructuredEmbedding):
    """ Reduce a list of embedded fields or messages using a feed forward. Requires list to be a constant size """

    def __init__(self, input_dims, encoding_output_dim, device):
        """

        :param input_dims: dimensions of the elements to reduce (list of size the number of elements, integer dimension).
        :param encoding_output_dim: dimension of the reduce output.
        """
        super().__init__(embedding_size=encoding_output_dim, device=device)
        self.encoding_dim = encoding_output_dim
        self.input_dims = input_dims
        sum_input_dims = 0
        for dim in input_dims:
            sum_input_dims += dim
        self.sum_input_dims = sum_input_dims
        self.linear1 = Linear(sum_input_dims, sum_input_dims * 2)
        self.linear2 = Linear(sum_input_dims * 2, encoding_output_dim)
        self.to(device)

    def forward(self, input_list, pad_missing=False):
        if len(input_list) != len(self.input_dims) and not pad_missing:
            raise ValueError("The input_list dimension does not match input_dims, and pad_missing is not specified. ")
        batch_size = input_list[0].size(0)
        if pad_missing:
            index = 0
            while len(input_list) < len(self.input_dims):
                # pad input list with minibatches of zeros:
                variable = torch.zeros(batch_size, self.input_dims[index]).to(self.device)

                input_list += [variable]
                index += 1

        if any([input.dim() != 2 for input in input_list]) or [input.size(1) for input in
                                                               input_list] != self.input_dims:
            print("STOP: input dims={} sizes={}".format([input.dim() for input in input_list],
                                                        [input.size() for input in input_list]))
        for index, input in enumerate(input_list):
            assert input.size(1) == self.input_dims[index], "input dimension must match declared for input {}. " \
                                                            "Was {}, should have been {}".format(
                index,input.size(1),self.input_dims[index] )
        x = torch.cat(input_list, dim=1)

        return self.forward_flat_inputs(x)

    def forward_flat_inputs(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        return x.view(-1, self.encoding_dim)


class Dispatcher():
    def __init__(self, mappers):
        self.mappers = mappers

    def dispatch(self, structure, ):
        type_ = structure['type']
        mapper = self.mappers[type_]

        result = mapper(structure)
        assert result is not None, "mapper for type {} returned None.".format(type_)
        return result

    def mapper_id(self, structure):
        """
        Return the id of the mapper that will process this instance.
        :param structure:
        :return:
        """
        type_ = structure['type']
        mapper = self.mappers[type_]
        return id(mapper)

    def mapper(self, structure):
        """
        Return the mapper that will process this instance.
        :param structure:
        :return:
        """
        type_ = structure['type']
        mapper = self.mappers[type_]
        return mapper

    def mapper_for_type(self, type_):
        """
        Return the mapper that will process this instance.
        :param structure:
        :return:
        """
        mapper = self.mappers[type_]
        return mapper


def store_indices_in_message(mapper, message, indices):
    mapper_id = id(mapper)
    if mapper_id not in message['indices']:
        message['indices'][mapper_id] = []
    message['indices'][mapper_id] += indices


def get_indices_in_message(mapper, message):
    return message['indices'][id(mapper)]


class BatchOfInstances(Module):
    """Takes a list of structure instances and produce a tensor of size batch x embedding dim of each instance."""

    def __init__(self, mappers, all_modules, device):
        """

        :param mappers: a dictionary, mapping message type to its mapper.
        """
        super().__init__()

        self.mappers = Dispatcher(mappers)
        self.all_modules = ModuleList(all_modules)
        self.to(device)

    def forward(self, instance_list):
        instance_list = list(instance_list)
        mapped = [self.mappers.dispatch(instance) for instance in instance_list]
        return torch.cat(mapped, dim=0)

    def preload(self, preloaded_tensor_list):
        mapped = [self.mappers.dispatch(instance) for instance in preloaded_tensor_list]
        return torch.cat(mapped, dim=0)