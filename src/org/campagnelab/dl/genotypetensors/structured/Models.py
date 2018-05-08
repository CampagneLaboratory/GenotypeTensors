import threading

import torch
from torch.autograd import Variable
from torch.nn import Module, Embedding, LSTM, Linear, ModuleList


class TensorCache:
    def __init__(self):
        self.cached_tensors = {}
        self.lock=threading.Lock()
        self.is_cuda=False
        self.device=-1

    def cuda(self,device=0):
        for key in self.cached_tensors.keys():
            tensor = self.cached_tensors[key]
            if not tensor.is_cuda:
                self.cached_tensors[key] = tensor.cuda(device)
        self.is_cuda=True
        self.device=device

    def cache(self, key, tensor_creation_lambda):
        if isinstance(key,list):
            internal_key=tuple(key)
        else:
            internal_key=key
        with self.lock:

            if internal_key in self.cached_tensors:
                tensor= self.cached_tensors[internal_key]
            else:
                tensor= tensor_creation_lambda(key)
                if self.is_cuda:
                    tensor=tensor.cuda(self.device)
                self.cached_tensors[internal_key] =tensor
            return Variable(tensor.data,requires_grad=tensor.requires_grad, volatile=tensor.volatile)
            #return tensor


class NoCache(TensorCache):
    def __init__(self):
        pass

    def cache(self, key, tensor_creation_lambda):
        return tensor_creation_lambda(key)


class StructuredEmbedding(Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

    def define_long_variable(self, values, cuda=None):
        variable = Variable(torch.LongTensor(values), requires_grad=False)
        if self.is_cuda(cuda):
            variable = variable.cuda(async=True)
        return variable

    def is_cuda(self, cuda=None):
        """ Return True iff this model is on cuda. """
        if cuda is not None:
            return cuda
        else:
            cuda = next(self.parameters()).data.is_cuda
        return cuda

    def collect_inputs(self,values,phase=0,cuda=None, batcher=None):
        pass

    def forward_batch(self,batcher, phase=0):
        pass

class map_Boolean(StructuredEmbedding):
    def __init__(self):
        super().__init__(2)
        self.true_value = Variable(torch.FloatTensor([[1, 0]]))
        self.false_value = Variable(torch.FloatTensor([[0, 1]]))


    def __call__(self, predicate, tensor_cache, cuda=None):

        return tensor_cache.cache(key=predicate, tensor_creation_lambda=
        lambda predicate: Variable(torch.FloatTensor([[1, 0]]))  if predicate else  Variable(torch.FloatTensor([[0, 1]])))
        #return Variable(value.data,requires_grad=True)
        #return value

class IntegerModel(StructuredEmbedding):
    def __init__(self, distinct_numbers, embedding_size):
        super().__init__(embedding_size)
        self.distinct_numbers = distinct_numbers
        self.embedding = Embedding(distinct_numbers, embedding_size)
        self.embedding.requires_grad = True

    def forward(self, values,tensor_cache, cuda=None):
        """Accepts a list of integer values and produces a batch of batch x embedded-value. """

        assert isinstance(values, list), "values must be a list of integers."
        for value in values:
            assert value < self.distinct_numbers, "A value is larger than the embedding input allow: " + str(value)
        if len(values)>4:

            # cache individual values for longer lists:
            cached_values=[]
            for value in values:
                cached_values+=[ tensor_cache.cache(key=[value],
                                            tensor_creation_lambda=lambda key: self.define_long_variable(key, cuda))]
            values=torch.cat(cached_values,dim=0)
        else:
            # cache the whole list for short ones:
            values=tensor_cache.cache(key=values, tensor_creation_lambda=lambda key: self.define_long_variable(key,cuda))
        return  self.embedding(values)

    def collect_inputs(self,values,cuda=None,phase=0, batcher=None):
        return self.define_long_variable(values, cuda)

    def forward_batch(self,batcher,phase=0):
        return self.embedding(batcher.get_batched_input(mapper=self))

class MeanOfList(Module):
    def __init__(self):
        super().__init__()

    def forward(self, list_of_embeddings):
        assert isinstance(list_of_embeddings, Variable) or isinstance(list_of_embeddings,
                                                                      torch.FloatTensor), "input must be a Variable or Tensor"
        num_elements = list_of_embeddings.size(0)
        return (torch.sum(list_of_embeddings, dim=0) / num_elements).view(1, -1)

class RNNOfList(StructuredEmbedding):
    def __init__(self, embedding_size, hidden_size, num_layers=1):
        super().__init__(hidden_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = LSTM(embedding_size, hidden_size, num_layers, batch_first=True)

    def forward(self, list_of_embeddings, cuda=None):

        batch_size = list_of_embeddings.size(0)
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        hidden = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        memory = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        if self.is_cuda(cuda):
            hidden = hidden.cuda(async=True)
            memory = memory.cuda(async=True)
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

        self.linear1 = Linear(sum_input_dims, sum_input_dims * 2)
        self.linear2 = Linear(sum_input_dims * 2, encoding_output_dim)

    def forward(self, input_list, cuda=None, pad_missing=False):
        if len(input_list) != len(self.input_dims) and not pad_missing:
            raise ValueError("The input_list dimension does not match input_dims, and pad_missing is not specified. ")
        batch_size = input_list[0].size(0)
        if pad_missing:
            index = 0
            while len(input_list) < len(self.input_dims):
                # pad input list with minibatches of zeros:
                variable = Variable(torch.zeros(batch_size, self.input_dims[index]))
                if self.is_cuda(cuda):
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
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)

        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        return x.view(input_list[0].size(0), self.encoding_dim)


class Dispatcher():
    def __init__(self, mappers):
        self.mappers = mappers

    def dispatch(self, structure, tensor_cache, cuda=None):
        type_ = structure['type']
        mapper = self.mappers[type_]

        result = mapper(structure, tensor_cache=tensor_cache, cuda=cuda)
        assert result is not None, "mapper for type {} returned None.".format(type_)
        return result

    def collect_inputs(self, structure, tensor_cache, cuda=None):
        type_ = structure['type']
        mapper = self.mappers[type_]

        inputs= mapper.collect_inputs(structure, tensor_cache=tensor_cache, cuda=cuda)
        assert inputs is not None, "collect_inputs for mapper associated with type {} returned None.".format(type_)

        return {type_:inputs}

    def forward_batch(self,batcher):
        """inputs contains one key per type of structure. """
        mapped_results={}
        inputs=batcher.get_batched_input(mapper=self)
        for type_ in inputs.keys():
            mapper = self.mappers[type_]

            mapped_results[type_] = mapper.forward_batch(inputs[type_])
        return mapped_results


class BatchOfInstances(Module):
    """Takes a list of structure instances and produce a tensor of size batch x embedding dim of each instance."""

    def __init__(self, mappers, all_modules):
        """

        :param mappers: a dictionary, mapping message type to its mapper.
        """
        super().__init__()
        self.mappers = Dispatcher(mappers)
        self.all_modules = ModuleList(all_modules)

    def forward(self, instance_list,tensor_cache, cuda=None):
        mapped = [self.mappers.dispatch(instance, tensor_cache=tensor_cache,cuda=cuda) for instance in instance_list]
        return torch.cat(mapped, dim=0)

    def collect_inputs(self,instance_list,cuda=None):
        input_list=[]
        for instance in instance_list:
            input_list+=[self.mapers.dispatch.collect_inputs(instance,cuda=cuda)]
        return input_list

    def forward_batch(self,batcher):
        inputs=batcher.get_batched_input(mapper=self)
        return self.mappers.forward_batch(torch.cat(inputs, dim=0))