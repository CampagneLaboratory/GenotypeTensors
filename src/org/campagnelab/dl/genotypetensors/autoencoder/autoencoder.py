from torch import nn


def create_autoencoder_model(model_name, problem, encoded_size=32):
    input_size=problem.input_size("input")
    encoder = nn.Sequential(nn.Linear(input_size, encoded_size), nn.Sigmoid())
    decoder = nn.Sequential(nn.Linear(encoded_size, input_size), nn.Sigmoid())
    autoencoder = nn.Sequential(encoder, decoder)
    return autoencoder