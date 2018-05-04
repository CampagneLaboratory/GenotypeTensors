import torch
from torch.autograd import Variable
from torch.nn import Module, Embedding, LSTM, Linear


class StructuredEmbedding(Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

class map_Boolean(StructuredEmbedding):
    def __init__(self):
        super().__init__(2)

    def __call__(self,predicate):
        return Variable(torch.FloatTensor([[1,0]]) if predicate else torch.FloatTensor([[0,1]]))

class IntegerModel(StructuredEmbedding):
    def __init__(self, distinct_numbers, embedding_size):
        super().__init__(embedding_size)
        self.embedding = Embedding(distinct_numbers, embedding_size)
        self.embedding.requires_grad=True

    def forward(self, values):
        """Accepts a list of integer values and produces a batch of batch x embedded-value. """
        assert isinstance(values,list), "values must be a list of integers."
        return self.embedding(Variable(torch.LongTensor(values),requires_grad=False))

class MeanOfList(Module):
    def __init__(self):
        super().__init__()

    def forward(self, list_of_embeddings):
        assert isinstance(list_of_embeddings,Variable) or isinstance(list_of_embeddings,torch.FloatTensor),"input must be a Variable or Tensor"
        num_elements=list_of_embeddings.size(0)
        return (torch.sum(list_of_embeddings,dim=0)/num_elements).view(1,-1)



class RNNOfList(StructuredEmbedding):
    def __init__(self,embedding_size,hidden_size,num_layers=1):
        super().__init__(hidden_size)
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.lstm=LSTM(embedding_size, hidden_size, num_layers, batch_first=True)

    def forward(self, list_of_embeddings):
        batch_size=list_of_embeddings.size(0)
        num_layers=self.num_layers
        hidden_size=self.hidden_size
        hidden=Variable(torch.zeros(num_layers, batch_size, hidden_size))
        memory=Variable(torch.zeros(num_layers, batch_size, hidden_size))
        states = (hidden,memory)
        inputs = list_of_embeddings.view(batch_size, 1, -1)
        out, states=self.lstm(inputs, states)
        # return the last output:
        all_outputs= out.squeeze()
        if (all_outputs.dim()>1):
            # return the last output:
            return all_outputs[-1,:]
        else:
            return all_outputs.view(1,-1)

class Reduce(Module):
    """ Reduce a list of embedded fields or messages using a feed forward. Requires list to be a constant size """
    def __init__(self,input_dims, encoding_output_dim):
        """

        :param input_dims: dimensions of the elements to reduce (list of size the number of elements, integer dimension).
        :param encoding_output_dim: dimension of the reduce output.
        """
        super().__init__()
        self.encoding_dim=encoding_output_dim
        self.input_dims=input_dims
        sum_input_dims=0
        for dim in input_dims:
            sum_input_dims+=dim

        self.linear1=Linear(sum_input_dims, sum_input_dims*2)
        self.linear2=Linear(sum_input_dims*2, encoding_output_dim)

    def forward(self, input_list):
        for index, input in enumerate(input_list):
            assert input.size(1)==self.input_dims[index], "input dimension must match declared for input {} ".format(index)
        x=torch.cat(input_list,dim=1)
        x=self.linear1(x)
        x=torch.nn.functional.relu(x)

        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        return x.view(input_list[0].size(0),self.encoding_dim)

class Dispatcher():
    def __init__(self, functions):
        self.functions=functions

    def dispatch(self, structure):
        type_ = structure['type']
        function=self.functions[type_]

        result= function(structure)
        assert result is not None, "mapper for type {} returned None.".format(type_)
        return result

class BatchOfInstances(Module):
    """Takes a list of structure instances and produce a tensor of size batch x embedding dim of each instance."""
    def __init__(self,mappers):
        """

        :param mappers: a dictionary, mapping message type to its mapper.
        """
        super().__init__()
        self.mappers=Dispatcher(mappers)

    def forward(self, instance_list):
        mapped=[self.mappers.dispatch(instance) for instance in instance_list]
        return torch.cat(mapped,dim=0)


