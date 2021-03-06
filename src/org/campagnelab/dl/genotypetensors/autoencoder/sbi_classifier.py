from torch import nn

from org.campagnelab.dl.genotypetensors.autoencoder.autoencoder import AutoEncoder
from org.campagnelab.dl.genotypetensors.autoencoder.autoencoder2 import AutoEncoder2
from org.campagnelab.dl.genotypetensors.autoencoder.semisup_adversarial_autoencoder import ConfigurableModule


class SbiSomaticClassifier(ConfigurableModule):
    """ A classifier that predicts is_mutated_output and somatic_frequency_output. forward returns a
    tuple of these predictions. """

    def __init__(self, input_size=32, num_layers=3, target_size=10, use_selu=False):
        super().__init__(use_selu=use_selu)
        layer_list = []
        for layer in range(0, num_layers):
            layer_list += [nn.Linear(input_size, input_size), self.non_linearity(), nn.Dropout()]

        self.features = nn.Sequential(*layer_list)
        self.is_mutated_linear = nn.Linear(input_size, target_size)
        self.freq_linear = nn.Linear(input_size, 1)

    def forward(self, input):
        """ Predicts (is_mutated_output, somatic_frequency_output) from the input. """
        features = self.features(input)
        is_mutated_output = self.is_mutated_linear(features)
        somatic_frequency_output = self.freq_linear(features)
        return (is_mutated_output, somatic_frequency_output)


class SbiGenotypeClassifier(ConfigurableModule):
    """ A classifier that predicts genotypes with a softmax. forward returns the output. """

    def __init__(self, input_size=32, num_layers=3, target_size=10, autoencoder=None,
                 dropout_p=0.2, prenormalized_inputs=False, use_selu=False):
        super().__init__(use_selu=use_selu)
        layer_list = []
        for layer in range(0, num_layers):
            layer_list += [nn.Dropout(dropout_p), nn.BatchNorm1d(input_size), nn.Linear(input_size, input_size),
                           self.non_linearity()]

        self.features = nn.Sequential(*layer_list)
        self.softmax_genotype_linear = nn.Sequential(nn.Dropout(dropout_p), nn.BatchNorm1d(input_size),
                                                     nn.Linear(input_size, target_size))
        self.autoencoder = autoencoder
        self.prenormalized_inputs = prenormalized_inputs

    def forward(self, input):
        """ Predicts (is_mutated_output, somatic_frequency_output) from the input. """
        encoded = self.autoencoder.encoder(input)
        features = self.features(encoded)
        softmax_genotype_linear = self.softmax_genotype_linear(features)
        return softmax_genotype_linear

    def get_auto_encoder(self):
        return self.autoencoder


def create_classifier_model(model_name, problem, encoded_size=32,
                            somatic=True, ngpus=1, dropout_p=0.2, num_layers=3,
                            autoencoder_type=2, drop_decoder=False, prenormalized_inputs=False,
                            use_selu=False):
    input_size = problem.input_size("input")

    assert len(input_size) == 1, "Classifier require 1D input features."
    autoencoder = AutoEncoder2(input_size=input_size[0], encoded_size=encoded_size, ngpus=ngpus,
                               dropout_p=dropout_p, use_selu=use_selu) \
        if autoencoder_type == 2 else \
        AutoEncoder(input_size=input_size[0], encoded_size=encoded_size, ngpus=ngpus, dropout_p=dropout_p,
                    use_selu=use_selu)

    if somatic:
        mut_base_size = problem.output_size("isBaseMutated")
        assert len(mut_base_size) == 1, "Classifier require 1D isBaseMutated features."
        # create an auto-encoder, we need the encoder part and discard the rest:

        del autoencoder.decode
        autoencoder.decoder = None
        classifier = SbiSomaticClassifier(input_size=encoded_size, target_size=mut_base_size[0], num_layers=num_layers)
        classifier = nn.Sequential(autoencoder.encoder, classifier)
    else:
        output_size = problem.output_size("softmaxGenotype")
        if drop_decoder:
            del autoencoder.decoder
            autoencoder.decoder = None
        # Store it in the classifier, so we can retrieve it for unsupervised reconstruction and to optimize its parameters:
        classifier = SbiGenotypeClassifier(input_size=encoded_size, target_size=output_size[0],
                                           autoencoder=autoencoder, dropout_p=dropout_p, num_layers=num_layers,
                                           prenormalized_inputs=prenormalized_inputs, use_selu=use_selu)
    print("classifier:" + str(classifier))
    return classifier
