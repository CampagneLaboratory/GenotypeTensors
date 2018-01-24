from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size=512, encoded_size=32):
        super().__init__()
        encoder_list = []
        encoder_input_size = input_size
        encoder_output_size = int(encoder_input_size / 2)
        while encoder_output_size > encoded_size:
            encoder_list += [nn.Linear(encoder_input_size, encoder_output_size), nn.ReLU()]

            encoder_input_size = encoder_output_size
            encoder_output_size = max(int(encoder_input_size / 2),encoded_size)

        encoder_list += [ nn.Linear(encoder_input_size, encoded_size) ]
        # Constrain the code towards a binary 1/0 code using a sigmoid:
        encoder_list += [ nn.Sigmoid()]
        encoder = nn.Sequential(*encoder_list)

        decoder_list = []
        decoder_input_size = encoded_size
        decoder_output_size = int(decoder_input_size * 2)
        while decoder_output_size < input_size:
            decoder_list += [ nn.Linear(decoder_input_size, decoder_output_size),  nn.ReLU()]

            decoder_input_size = decoder_output_size
            decoder_output_size = min(int(decoder_output_size * 2),input_size)

        decoder_list += [ nn.Linear(decoder_input_size,input_size), nn.ReLU()]

        decoder = nn.Sequential(*decoder_list)

        self.autoencoder = nn.Sequential( encoder,  decoder)

    def forward(self, input):
        return self.autoencoder(input)


def create_autoencoder_model(model_name, problem, encoded_size=32):
    input_size = problem.input_size("input")
    autoencoder = AutoEncoder(input_size=input_size, encoded_size=encoded_size)
    print("model" +str(autoencoder))
    return autoencoder
