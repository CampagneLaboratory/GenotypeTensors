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

    def cache(self, value, tensor_creation_lambda, key=None):
            if key is None:
                key=value
            internal_key=key
            if internal_key in self.cached_tensors:
                tensor= self.cached_tensors[internal_key]
            else:
                tensor= tensor_creation_lambda(value)
                if self.is_cuda:
                    tensor=tensor.cuda(self.device)
                self.cached_tensors[internal_key] =tensor
            return Variable(tensor.data,requires_grad=tensor.requires_grad, volatile=tensor.volatile)
            #return tensor


class NoCache(TensorCache):
    def __init__(self):
        pass

    def cache(self, value, tensor_creation_lambda,key=None ):
        if key is None:
            key = value
        return tensor_creation_lambda(value)


class StructuredEmbedding(Module):
    def __init__(self, embedding_size, use_cuda):
        super().__init__()
        self.embedding_size = embedding_size
        self.use_cuda=use_cuda

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

    def collect_inputs(self,values,phase=0,tensor_cache=NoCache(),cuda=None, batcher=None):
        """
        Obtain inputs from instances (values). The method will be called for values in a set until all inputs are
        collected for the set. After all inputs are collected, a forward batch call can be made on the mapper to
        calculate the forward of the batch. If additional mapping needs to be performed from the batched result, a
        second phase of collect_inputs and forward_batch may be done by calling these methods with phase=1, and so
        on as needed.
        :param values: instances to map. Accepts a list of instances for convenience.
        :param phase: An integer, 0 for first phase, incremental values for additional phases.
        :param tensor_cache: A cache to obtain new tensors.
        :param cuda: Whether the tensors must be moved to CUDA (default: no).
        :param batcher: The batcher can be used to retrieve batched results or inputs produced at an earlier phase.
        :return: A dict where keys are the id of the mappers that produced the input. Each value in the dict must be
            a list of inputs (Variables).
        """
        pass

    def forward_batch(self,batcher, phase=0):
        """
        Do a forward on batch. Will be called once per mapper per phase.
        :param batcher: The batcher holds the batches. Use get_batched_input to obtain a particular batch.
        :param phase: An integer, from 0 to n, indicating the phase of mapping.
        :return: The batched result of the forward phase, for an entire set of instances.
        """
        pass

class map_Boolean(StructuredEmbedding):
    def __init__(self,use_cuda):
        super().__init__(2,use_cuda)
        self.true_value = Variable(torch.FloatTensor([[1, 0]]))
        self.false_value = Variable(torch.FloatTensor([[0, 1]]))

    def forward(self, predicate, tensor_cache, cuda=None):
        assert isinstance(predicate,bool),"predicate must be a boolean"
        value= tensor_cache.cache(key=predicate,value=predicate, tensor_creation_lambda=
        lambda predicate: Variable(torch.FloatTensor([[1, 0]]))  if predicate else  Variable(torch.FloatTensor([[0, 1]])))
        if self.use_cuda:
            value=value.cuda(async=True)
        return value
        #return Variable(value.data,requires_grad=True)
        #return value

    def collect_inputs(self, values, tensor_cache=NoCache(), phase=0, cuda=None, batcher=None):
        if isinstance(values,list):
            cat = torch.cat([self(predicate, tensor_cache, cuda) for predicate in values], dim=1)
            if self.use_cuda:
                cat=cat.cuda(async=True)
            return batcher.store_inputs(mapper=self, inputs=cat)

        else:
            # only one value:
            return batcher.store_inputs(mapper=self, inputs= self(values, tensor_cache, cuda))

    def forward_batch(self, batcher, phase=0):

        result= batcher.get_batched_input(mapper=self)
        batcher.store_batched_result(mapper=self,batched_result=result)
        if isinstance(result,dict) and id(self) in result.keys():
            return result[id(self)]
        else:
            return result

class IntegerModel(StructuredEmbedding):
    def __init__(self, distinct_numbers, embedding_size,use_cuda):
        super().__init__(embedding_size,use_cuda)
        self.distinct_numbers = distinct_numbers
        self.embedding = Embedding(distinct_numbers, embedding_size)
        self.embedding.requires_grad = True

    def forward(self, values,tensor_cache=NoCache(), cuda=None):
        """Accepts a list of integer values and produces a batch of batch x embedded-value. """

        assert isinstance(values, list), "values must be a list of integers."
        for value in values:
            assert value < self.distinct_numbers, "A value is larger than the embedding input allow: " + str(value)

        # cache the embedded values:
        cached_values=[]
        for value in values:
            cached_values+=[
                tensor_cache.cache(key=(id(self),value),value=value, # one embedded value per IntegerModel and base.
                    tensor_creation_lambda=lambda value: self.embedding(self.define_long_variable([value], cuda)))
            ]
        values=torch.cat(cached_values,dim=0)
        return values

    def collect_inputs(self,values,phase=0,tensor_cache=NoCache(),cuda=None, batcher=None):
        return batcher.store_inputs(mapper=self,inputs=self.define_long_variable(values, cuda))

    def forward_batch(self,batcher,phase=0):
        batched= self.embedding(batcher.get_batched_input(mapper=self))
        batcher.store_batched_result(mapper=self,batched_result=batched)
        return batched

class MeanOfList(Module):
    def __init__(self):
        super().__init__()

    def forward(self, list_of_embeddings,cuda=None):
        assert isinstance(list_of_embeddings, Variable) or isinstance(list_of_embeddings,
                                                                      torch.FloatTensor), "input must be a Variable or Tensor"
        num_elements = list_of_embeddings.size(0)
        return (torch.sum(list_of_embeddings, dim=0) / num_elements).view(1, -1)

class RNNOfList(StructuredEmbedding):
    def __init__(self, embedding_size, hidden_size, num_layers=1,use_cuda=None):
        super().__init__(hidden_size,use_cuda)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        if use_cuda:
            self.lstm=self.lstm.cuda()

    def forward(self, list_of_embeddings, cuda=None):

        batch_size = list_of_embeddings.size(0)
        num_layers = self.num_layers
        hidden_size = self.hidden_size
        hidden = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        memory = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        if self.use_cuda:
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

    def __init__(self, input_dims, encoding_output_dim,use_cuda):
        """

        :param input_dims: dimensions of the elements to reduce (list of size the number of elements, integer dimension).
        :param encoding_output_dim: dimension of the reduce output.
        """
        super().__init__(embedding_size=encoding_output_dim,use_cuda=use_cuda)
        self.encoding_dim = encoding_output_dim
        self.input_dims = input_dims
        sum_input_dims = 0
        for dim in input_dims:
            sum_input_dims += dim
        self.sum_input_dims=sum_input_dims
        self.linear1 = Linear(sum_input_dims, sum_input_dims * 2)
        self.linear2 = Linear(sum_input_dims * 2, encoding_output_dim)
        if self.use_cuda:
            self.linear1=self.linear1.cuda()
            self.linear2=self.linear2.cuda()

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

        return self.forward_flat_inputs( x)

    def forward_flat_inputs(self,  x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        return x.view(-1, self.encoding_dim)

    def collect_inputs(self, x, phase=0, tensor_cache=NoCache(), cuda=None, batcher=None):
        assert x.size(-1)==self.sum_input_dims,"you must provided a pre-padded input to Reduce in batch mode."
        return batcher.store_inputs(self,x)

    def forward_batch(self, batcher, phase=0):
        return self.forward_flat_inputs(batcher.get_batched_input(self))


class Dispatcher():
    def __init__(self, mappers):
        self.mappers = mappers

    def dispatch(self, structure, tensor_cache=NoCache(), cuda=None):
        type_ = structure['type']
        mapper = self.mappers[type_]

        result = mapper(structure, tensor_cache=tensor_cache, cuda=cuda)
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

    def collect_inputs(self, structure, batcher, tensor_cache=NoCache(),phase=0, cuda=None):
        type_ = structure['type']
        mapper = self.mappers[type_]

        indices= mapper.collect_inputs(structure, phase,batcher=batcher, tensor_cache=tensor_cache, cuda=cuda)

        return indices

    def forward_batch(self,batcher,phase=0):
        """inputs contains one key per type of structure. """
        mapped_results={}
        inputs=batcher.get_batched_input(mapper=self)
        for type_ in inputs.keys():
            mapper = self.mappers[type_]

            mapped_results[type_] = mapper.forward_batch(inputs[type_],phase=0)
        return mapped_results


def store_indices_in_message(mapper, message, indices):
    mapper_id = id(mapper)
    if mapper_id not in message['indices']:
        message['indices'][mapper_id]= []
    message['indices'][mapper_id] += indices


def get_indices_in_message(mapper, message):
    return message['indices'][id(mapper)]


class BatchOfInstances(Module):
    """Takes a list of structure instances and produce a tensor of size batch x embedding dim of each instance."""

    def __init__(self, mappers, all_modules):
        """

        :param mappers: a dictionary, mapping message type to its mapper.
        """
        super().__init__()
        self.mappers = Dispatcher(mappers)
        self.all_modules = ModuleList(all_modules)

    def forward(self, instance_list, tensor_cache=NoCache(), cuda=None):
        instance_list = list(instance_list)
        mapped = [self.mappers.dispatch(instance, tensor_cache=tensor_cache,cuda=cuda) for instance in instance_list]
        return torch.cat(mapped, dim=0)

    def collect_inputs(self, instance_list, batcher, phase=0, tensor_cache=NoCache(), cuda=None):
        if phase == 0:
            for instance in instance_list:
                if 'indices' not in instance:
                    instance['indices'] = {}
                    store_indices_in_message(mapper=self.mappers.mapper(instance), message=instance,
                                             indices=self.mappers.collect_inputs(structure=instance, phase=phase,
                                                                                 cuda=cuda, batcher=batcher,
                                                                                 tensor_cache=tensor_cache))

            return None

    def forward_batch(self, batcher, phase=0):
        inputs = batcher.get_batched_input(mapper=self)
        return self.mappers.forward_batch(torch.cat(inputs, dim=0), phase=0)

