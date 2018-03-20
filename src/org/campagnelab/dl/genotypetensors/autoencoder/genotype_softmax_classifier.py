import math

from torch import nn

from org.campagnelab.dl.genotypetensors.autoencoder.semisup_adversarial_autoencoder import ConfigurableModule


class GenotypeSoftmaxClassifer(ConfigurableModule):
    def __init__(self, num_inputs, target_size, num_layers, reduction_rate, model_capacity,
                 dropout_p, ngpus, use_selu=False, skip_batch_norm=False, add_softmax=False):
        super().__init__(use_selu=use_selu)
        layer_list = []
        num_in = num_inputs
        num_hidden_nodes = int(num_inputs * model_capacity)
        for layer_idx in range(1, num_layers + 1):
            num_out = int(num_hidden_nodes * math.pow(reduction_rate, layer_idx) * model_capacity)
            if not skip_batch_norm:
                layer_list += [nn.Dropout(dropout_p), nn.BatchNorm1d(num_in), nn.Linear(num_in, num_out),
                               self.non_linearity()]
            else:
                layer_list += [nn.Dropout(dropout_p), nn.Linear(num_in, num_out), self.non_linearity()]
            num_in = num_out
        self.features = nn.Sequential(*layer_list)
        if not skip_batch_norm:
            if add_softmax:
                self.softmax_genotype_linear = nn.Sequential(nn.Dropout(dropout_p), nn.BatchNorm1d(num_in),
                                                             nn.Linear(num_in, target_size), nn.Softmax(0))
            else:
                self.softmax_genotype_linear = nn.Sequential(nn.Dropout(dropout_p), nn.BatchNorm1d(num_in),
                                                             nn.Linear(num_in, target_size))
        else:
            if add_softmax:
                self.softmax_genotype_linear = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(num_in, target_size),
                                                             nn.Softmax(0))
            else:
                self.softmax_genotype_linear = nn.Sequential(nn.Dropout(dropout_p), nn.Linear(num_in, target_size))
        self.ngpus = ngpus
        self.device_list = list(range(0, self.ngpus))

    def non_linearity(self):
        return super().non_linearity()

    def forward(self, model_input):
        if self.ngpus > 1 and model_input.data.is_cuda():
            features = nn.parallel.data_parallel(self.features, model_input, self.device_list)
            softmax_genotype_linear = nn.parallel.data_parallel(self.softmax_genotype_linear, features,
                                                                self.device_list)
        else:
            features = self.features(model_input)
            softmax_genotype_linear = self.softmax_genotype_linear(features)
        return softmax_genotype_linear


def create_genotype_softmax_classifier_model(model_name, problem, ngpus, num_layers, reduction_rate, model_capacity,
                                             dropout_p, use_selu, skip_batch_norm, add_softmax):
    input_size = problem.input_size("input")
    output_size = problem.output_size("softmaxGenotype")
    classifier = GenotypeSoftmaxClassifer(num_inputs=input_size[0], target_size=output_size[0], num_layers=num_layers,
                                          reduction_rate=reduction_rate, model_capacity=model_capacity,
                                          dropout_p=dropout_p, ngpus=ngpus, use_selu=use_selu,
                                          skip_batch_norm=skip_batch_norm, add_softmax=add_softmax)
    print("classifier:" + str(classifier))
    return classifier
