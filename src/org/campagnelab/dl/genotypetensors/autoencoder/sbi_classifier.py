from torch import nn

from org.campagnelab.dl.genotypetensors.autoencoder.autoencoder import AutoEncoder


class SbiSomaticClassifier(nn.Module):
    """ A classifier that predicts is_mutated_output and somatic_frequency_output. forward returns a
    tuple of these predictions. """
    def __init__(self, input_size=32, num_layers=3, target_size=10):
        super().__init__()
        layer_list = []
        for layer in range(0,num_layers):
            layer_list += [nn.Linear(input_size, input_size), nn.ReLU(), nn.Dropout()]

        self.features = nn.Sequential(*layer_list)
        self.is_mutated_linear=nn.Linear(input_size,target_size)
        self.freq_linear=nn.Linear(input_size,1)

    def forward(self, input):
        """ Predicts (is_mutated_output, somatic_frequency_output) from the input. """
        features = self.features(input)
        is_mutated_output= self.is_mutated_linear(features)
        somatic_frequency_output= self.freq_linear(features)
        return (is_mutated_output, somatic_frequency_output)


def create_classifier_model(model_name, problem, encoded_size=32, somatic=True):
    input_size = problem.input_size("input")
    mut_base_size = problem.output_size("isBaseMutated")
    assert len(input_size)==1, "Classifier require 1D input features."
    assert len(mut_base_size)==1, "Classifier require 1D isBaseMutated features."
    if somatic:
        # create an auto-encoder, we need the encoder part and discard the rest:
        autoencoder = AutoEncoder(input_size=input_size[0], encoded_size=encoded_size)
        autoencoder.decoder=None
        classifier = SbiSomaticClassifier(input_size=encoded_size, target_size=mut_base_size[0])
        classifier=nn.Sequential(autoencoder.encoder, classifier)
    print("classifier:" +str(classifier))
    return classifier