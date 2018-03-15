from torch import nn
from torch.nn import BatchNorm1d

from org.campagnelab.dl.genotypetensors.autoencoder.semisup_adversarial_autoencoder import ConfigurableModule


class AutoEncoder2(ConfigurableModule):
    def __init__(self, input_size=512, encoded_size=32, ngpus=1, dropout_p=0,
                 nreplicas=0,use_selu=False):
        """

        :param input_size:
        :param encoded_size:
        :param ngpus: Number of gpus to run on.
        :param nreplicas: Number of model replica per GPUs, if ngpu>1
        """
        super().__init__(use_selu=use_selu)
        self.ngpu = ngpus
        self.device_list = list(range(0, ngpus))
        encoder_list = []
        encoder_input_size = input_size
        encoder_output_size = int(encoder_input_size / 2)
        encoder_list += [nn.Dropout(dropout_p),
                         nn.BatchNorm1d(encoder_input_size), nn.Linear(input_size, input_size * 2),
                         nn.Dropout(0.5),
                         nn.BatchNorm1d(input_size * 2), nn.Linear(input_size * 2, input_size)]
        while encoder_output_size > encoded_size:
            encoder_list += [nn.BatchNorm1d(encoder_input_size), nn.Linear(encoder_input_size, encoder_output_size)]

            encoder_input_size = encoder_output_size
            encoder_output_size = max(int(encoder_input_size / 2), encoded_size)

        encoder_list += [nn.BatchNorm1d(encoder_input_size), nn.Linear(encoder_input_size, encoded_size)]
        # Constrain the code towards a binary 1/0 code using a sigmoid:
        # encoder_list += [ nn.Sigmoid()]
        self.encoder = nn.Sequential(*encoder_list)

        decoder_list = []
        decoder_input_size = encoded_size
        decoder_output_size = int(decoder_input_size * 2)
        while decoder_output_size < input_size:
            decoder_list += [nn.BatchNorm1d(decoder_input_size), nn.Linear(decoder_input_size, decoder_output_size)]
            decoder_input_size = decoder_output_size
            decoder_output_size = min(int(decoder_output_size * 2), input_size)

        decoder_list += [nn.BatchNorm1d(decoder_input_size), nn.Linear(decoder_input_size, input_size)]

        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpu > 1:
            encoded = nn.parallel.data_parallel(self.encoder, model_input, self.device_list)
            decoded = nn.parallel.data_parallel(self.decoder, encoded, self.device_list)
        else:
            encoded = self.encoder(input)
            decoded = self.decoder(encoded)
        return decoded


def create_autoencoder_model(model_name, problem, encoded_size=32, ngpus=1, nreplicas=1, dropout_p=0):
    input_size = problem.input_size("input")
    assert len(input_size) == 1, "AutoEncoders required 1D input features."

    autoencoder = AutoEncoder2(input_size=input_size[0], encoded_size=encoded_size, ngpus=ngpus, nreplicas=nreplicas,
                               dropout_p=dropout_p)
    print("model" + str(autoencoder))
    return autoencoder
