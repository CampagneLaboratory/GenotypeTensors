import numpy
import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import MSELoss, MultiLabelSoftMarginLoss, Linear

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import recode_for_label_smoothing
from org.campagnelab.dl.utils.utils import draw_from_categorical, draw_from_gaussian

class ConfigurableModule(nn.Module):
    def __init__(self, use_selu=False):

        super().__init__()
        self.use_selu=use_selu

    def non_linearity(self):
        if self.use_selu:
            return nn.SELU()
        else:
            return nn.ReLU()

class _SemiSupAdvEncoder(ConfigurableModule):
    """The encoder takes inputs and maps them to the latent space (of dimension latent_code_dim). """
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10,
                 latent_code_dim=2, use_selu=False):
        super().__init__(use_selu=use_selu)
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        # Create encoders
        encoder_list = [
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, n_dim),
            nn.Dropout(dropout_p),
            self.non_linearity()
        ]
        for _ in range(num_hidden_layers):
            encoder_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                self.non_linearity()
            ]
        self.encoder_base = nn.Sequential(*encoder_list)
        self.latent_encoder = nn.Linear(n_dim, latent_code_dim)
        self.category_encoder = nn.Linear(n_dim, num_classes)



    def forward(self, model_input):
        """Accepts inputs and produces a tuple of predicted categories and latent code."""
        if model_input.data.is_cuda and self.ngpus > 1:
            base = nn.parallel.data_parallel(self.encoder_base, model_input, self.device_list)
            latent_code = nn.parallel.data_parallel(self.latent_encoder, base, self.device_list)
            category_encoded = nn.parallel.data_parallel(self.category_encoder, base, self.device_list)
        else:
            base = self.encoder_base(model_input)
            latent_code = self.latent_encoder(base)
            category_encoded = self.category_encoder(base)
        return category_encoded, latent_code


class _SemiSupAdvDecoder(ConfigurableModule):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10,
                 prior_dim=2,use_selu=False):
        super().__init__(use_selu=use_selu)
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        # Create decoder
        decoder_list = [
            nn.BatchNorm1d(prior_dim + num_classes),
            nn.Linear(prior_dim + num_classes, n_dim),
            nn.Dropout(dropout_p),
            self.non_linearity(),
        ]
        for _ in range(num_hidden_layers):
            decoder_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                self.non_linearity()
            ]
        decoder_list += [
            nn.BatchNorm1d(n_dim),
            nn.Linear(n_dim, input_size)
        ]
        self.decoder = nn.Sequential(*decoder_list)
    
    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpus > 1:
            decoded = nn.parallel.data_parallel(self.decoder, model_input, self.device_list)
        else:
            decoded = self.decoder(model_input)
        return decoded


class _SemiSupAdvDiscriminatorCat(ConfigurableModule):
    def __init__(self, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3,
                 num_classes=10,use_selu=False):
        super().__init__(use_selu=use_selu)
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))
        self.num_classes=num_classes

        layer_list = [
            nn.BatchNorm1d(num_classes),
            nn.Linear(num_classes, n_dim),
            nn.Dropout(dropout_p),
            self.non_linearity()
        ]

        for _ in range(num_hidden_layers):
            layer_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                self.non_linearity()
            ]
        
        layer_list += [
            nn.BatchNorm1d(n_dim),
            nn.Linear(n_dim, 1),
            nn.Sigmoid()
        ]
        self.cat_discriminator = nn.Sequential(*layer_list)
    
    def forward(self, categories):
        if categories.data.is_cuda and self.ngpus > 1:
            category = nn.parallel.data_parallel(self.cat_discriminator, categories, self.device_list)
        else:
            category = self.cat_discriminator(categories)
        return category


class _SemiSupAdvDiscriminatorPrior(ConfigurableModule):
    def __init__(self, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3,
                 prior_dim=2,use_selu=False):
        super().__init__(use_selu=use_selu)
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        layer_list = [
            nn.BatchNorm1d(prior_dim),
            nn.Linear(prior_dim, n_dim),
            nn.Dropout(dropout_p),
            self.non_linearity()
        ]

        for _ in range(num_hidden_layers):
            layer_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                self.non_linearity()
            ]
        
        layer_list += [
            nn.BatchNorm1d(n_dim),
            nn.Linear(n_dim, 1),
            nn.Sigmoid()
        ]
        self.prior_discriminator = nn.Sequential(*layer_list)
    
    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpus > 1:
            prior_val = nn.parallel.data_parallel(self.prior_discriminator, model_input, self.device_list)
        else:
            prior_val = self.prior_discriminator(model_input)
        return prior_val


class SemiSupAdvAutoencoder(ConfigurableModule):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10, prior_dim=2,
                 epsilon=1E-15, seed=None, mini_batch=7, prenormalized_inputs=False,
                 use_selu=False):
        """

        :param input_size:
        :param n_dim:
        :param ngpus:
        :param dropout_p:
        :param num_hidden_layers:
        :param num_classes:
        :param prior_dim:
        :param epsilon:
        :param seed:
        :param mini_batch:
        :param prenormalized_inputs: True when the inputs must be normalized by mean and std before using this model.
        """
        super().__init__(use_selu=use_selu)
        self.epsilon = epsilon
        self.prior_dim = prior_dim
        self.num_classes = num_classes
        self.seed = seed
        self.reconstruction_criterion = MSELoss()
        self.semisup_loss_criterion = MultiLabelSoftMarginLoss()
        self.categorical_distribution = None
        self.prenormalized_inputs=prenormalized_inputs
        self.encoder = _SemiSupAdvEncoder(input_size=input_size, n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                          num_hidden_layers=num_hidden_layers, num_classes=num_classes,
                                          latent_code_dim=prior_dim,use_selu=use_selu)
        self.decoder = _SemiSupAdvDecoder(input_size=input_size, n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                          num_hidden_layers=num_hidden_layers, num_classes=num_classes,
                                          prior_dim=prior_dim,use_selu=use_selu)
        self.discriminator_cat = _SemiSupAdvDiscriminatorCat(n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                                             num_hidden_layers=num_hidden_layers,
                                                             num_classes=num_classes,use_selu=use_selu)
        self.discriminator_prior = _SemiSupAdvDiscriminatorPrior(n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                                                 num_hidden_layers=num_hidden_layers,
                                                                 prior_dim=prior_dim,use_selu=use_selu)



    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.semisup_loss_criterion = MultiLabelSoftMarginLoss()

    def forward(self, model_input):
        """Return the predicted category (as one hot encoding)."""
        category_code, latent_code = self.encoder(model_input)
        return category_code


    def _get_concat_code(self, model_input, target=None):
        category_code, latent_code = self.encoder(model_input)
        if target is not None:
            category_code=target
        return torch.cat([category_code, latent_code], 1)

    def get_reconstruction_loss(self, model_input):

        latent_code = self._get_concat_code(model_input)
        reconstructed_model_input = self.decoder(latent_code)
        model_output=Variable(model_input.data + self.epsilon, requires_grad=False)
        reconstruction_loss = self.reconstruction_criterion(reconstructed_model_input + self.epsilon, model_output)

        return reconstruction_loss

    def get_crossconstruction_loss(self, model_input1, model_input2, target2):

        latent_code = self._get_concat_code(model_input1, target2)
        reconstructed_model_input = self.decoder(latent_code)
        model_output=Variable(model_input2.data + self.epsilon, requires_grad=False)
        reconstruction_loss = self.reconstruction_criterion(reconstructed_model_input + self.epsilon, model_output)

        return reconstruction_loss

    def get_discriminator_loss(self, common_trainer, model_input, category_prior=None, recode_labels=True):

        mini_batch_size = model_input.data.size()[0]
        categories_real = common_trainer.dreamup_target_for(input=model_input, num_classes=self.num_classes,
                                                            category_prior=category_prior).to(self.device)
        prior_real = draw_from_gaussian(self.prior_dim, mini_batch_size).to(self.device)
        categories_input, prior_input = self.encoder(model_input)
        cat_prob_real = self.discriminator_cat(categories_real)
        cat_prob_input = self.discriminator_cat(categories_input)
        prior_prob_real = self.discriminator_prior(prior_real)
        prior_prob_input = self.discriminator_prior(prior_input)
        cat_discriminator_loss = -torch.mean(
            torch.log(cat_prob_real + self.epsilon) + torch.log(1 - cat_prob_input + self.epsilon)
        )
        prior_discriminator_loss = -torch.mean(
            torch.log(prior_prob_real + self.epsilon) + torch.log(1 - prior_prob_input + self.epsilon)
        )
        discriminator_loss = cat_discriminator_loss + prior_discriminator_loss


        return discriminator_loss

    def get_generator_loss(self, model_input):

        categories_input, hidden_state = self.encoder(model_input)
        cat_prob_input = self.discriminator_cat(categories_input)
        prior_prob_input = self.discriminator_prior(hidden_state)
        generator_loss =  -torch.mean(torch.log(cat_prob_input + self.epsilon))
        generator_loss += -torch.mean(torch.log(prior_prob_input + self.epsilon))

        return generator_loss

    def get_semisup_loss(self, model_input, categories_target):

        categories_input, _ = self.encoder(model_input)
        semisup_loss = self.semisup_loss_criterion(categories_input, categories_target)
        return semisup_loss

    def get_crossencoder_supervised_loss(self, model_input, categories_target):

        predicted_categories, latent_code = self.encoder(model_input)
        supervised_loss = self.semisup_loss_criterion(predicted_categories, categories_target)
        return supervised_loss


def create_semisup_adv_autoencoder_model(model_name, problem, encoded_size=32, ngpus=1, nreplicas=1, dropout_p=0,
                                         n_dim=500, num_hidden_layers=1,prenormalized_inputs=False,
                                         use_selu=False):
    input_size = problem.input_size("input")
    num_classes = problem.output_size("softmaxGenotype")
    assert len(input_size) == 1, "AutoEncoders require 1D input features."
    assert len(num_classes) == 1, "AutoEncoders require 1D output features."
    semisup_adv_autoencoder = SemiSupAdvAutoencoder(input_size=input_size[0], n_dim=n_dim, ngpus=ngpus,
                                                    dropout_p=dropout_p, num_hidden_layers=num_hidden_layers,
                                                    num_classes=num_classes[0], prior_dim=encoded_size,
                                                    prenormalized_inputs=prenormalized_inputs,
                                                    use_selu=use_selu)
    print(semisup_adv_autoencoder)
    return semisup_adv_autoencoder
