from torch import nn
from torch.nn import BatchNorm1d


class AutoEncoder(nn.Module):
    def __init__(self, input_size=512, encoded_size=32, ngpus=1):
        """

        :param input_size:
        :param encoded_size:
        :param ngpu: Number of gpus to run on.
        :param num_replica: Number of model replica per GPUs, if ngpu>1
        """
        super().__init__()
        self.ngpu=ngpus
        self.device_list=list(range(0,ngpus))
        encoder_list = []
        encoder_input_size = input_size
        encoder_output_size = int(encoder_input_size / 2)
        while encoder_output_size > encoded_size:
            encoder_list += [ nn.BatchNorm1d(encoder_input_size), nn.Linear(encoder_input_size, encoder_output_size), nn.SELU()]

            encoder_input_size = encoder_output_size
            encoder_output_size = max(int(encoder_input_size / 2),encoded_size)

        encoder_list += [ nn.Linear(encoder_input_size, encoded_size) , nn.SELU()]
        # Constrain the code towards a binary 1/0 code using a sigmoid:
        encoder_list += [ nn.Sigmoid()]
        self.encoder = nn.Sequential(*encoder_list)

        decoder_list = []
        decoder_input_size = encoded_size
        decoder_output_size = int(decoder_input_size * 2)
        while decoder_output_size < input_size:
            decoder_list += [ nn.BatchNorm1d(encoder_input_size), nn.Linear(decoder_input_size, decoder_output_size),  nn.SELU()]
            decoder_input_size = decoder_output_size
            decoder_output_size = min(int(decoder_output_size * 2),input_size)

        decoder_list += [ nn.Linear(decoder_input_size,input_size), nn.SELU()]

        self.decoder = nn.Sequential(*decoder_list)



    def forward(self, input):
        if input.data.is_cuda and self.ngpu > 1:
            encoded = nn.parallel.data_parallel(self.encoder, input, self.device_list)
            decoded = nn.parallel.data_parallel(self.decoder, encoded, self.device_list)
        else:
            encoded=self.encoder(input)
            decoded=self.decoder(encoded)
        return decoded


def create_autoencoder_model(model_name, problem, encoded_size=32, ngpus=1,nreplicas=1):
    input_size = problem.input_size("input")
    assert len(input_size)==1, "AutoEncoders required 1D input features."

    autoencoder = AutoEncoder(input_size=input_size[0], encoded_size=encoded_size,ngpus=ngpus,nreplicas=nreplicas)
    print("model" +str(autoencoder))
    return autoencoder
