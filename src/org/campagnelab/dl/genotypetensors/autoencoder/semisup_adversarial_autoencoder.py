import numpy
import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical

from org.campagnelab.dl.utils.utils import draw_from_categorical, draw_from_gaussian


class _SemiSupAdvEncoder(nn.Module):
    """The encoder takes inputs and maps them to the latent space (of dimension latent_code_dim). """
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10,
                 latent_code_dim=2):
        super().__init__()
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        # Create encoders
        encoder_list = [
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, n_dim),
            nn.Dropout(dropout_p),
            nn.SELU()
        ]
        for _ in range(num_hidden_layers):
            encoder_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                nn.SELU()
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
        return  category_encoded, latent_code


class _SemiSupAdvDecoder(nn.Module):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10,
                 prior_dim=2):
        super().__init__()
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        # Create decoder
        decoder_list = [
            nn.BatchNorm1d(prior_dim + num_classes),
            nn.Linear(prior_dim + num_classes, n_dim),
            nn.Dropout(dropout_p),
            nn.SELU(),
        ]
        for _ in range(num_hidden_layers):
            decoder_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                nn.SELU()
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


class _SemiSupAdvDiscriminatorCat(nn.Module):
    def __init__(self, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10):
        super().__init__()
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))
        self.num_classes=num_classes

        layer_list = [
            nn.BatchNorm1d(num_classes),
            nn.Linear(num_classes, n_dim),
            nn.Dropout(dropout_p),
            nn.SELU()
        ]

        for _ in range(num_hidden_layers):
            layer_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                nn.SELU()
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


class _SemiSupAdvDiscriminatorPrior(nn.Module):
    def __init__(self, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, prior_dim=2):
        super().__init__()
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        layer_list = [
            nn.BatchNorm1d(prior_dim),
            nn.Linear(prior_dim, n_dim),
            nn.Dropout(dropout_p),
            nn.SELU()
        ]

        for _ in range(num_hidden_layers):
            layer_list += [
                nn.BatchNorm1d(n_dim),
                nn.Linear(n_dim, n_dim),
                nn.Dropout(dropout_p),
                nn.SELU()
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


class SemiSupAdvAutoencoder(nn.Module):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10, prior_dim=2,
                 delta=1E-15, seed=None, mini_batch=7):
        super().__init__()
        self.encoder = _SemiSupAdvEncoder(input_size=input_size, n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                          num_hidden_layers=num_hidden_layers, num_classes=num_classes,
                                          latent_code_dim=prior_dim)
        self.decoder = _SemiSupAdvDecoder(input_size=input_size, n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                          num_hidden_layers=num_hidden_layers, num_classes=num_classes,
                                          prior_dim=prior_dim)
        self.discriminator_cat = _SemiSupAdvDiscriminatorCat(n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                                             num_hidden_layers=num_hidden_layers,
                                                             num_classes=num_classes)
        self.discriminator_prior = _SemiSupAdvDiscriminatorPrior(n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                                                 num_hidden_layers=num_hidden_layers,
                                                                 prior_dim=prior_dim)
        self.delta = delta
        self.prior_dim = prior_dim
        self.num_classes = num_classes
        self.seed = seed
        self.categorical_distribution=None

    def forward(self, model_input):
        return self.decoder(self._get_concat_code(model_input))

    def _get_concat_code(self, model_input):
        category_code, latent_code = self.encoder(model_input)
        return torch.cat([category_code, latent_code], 1)

    def get_reconstruction_loss(self, model_input, criterion, *opts):
        self.decoder.train()
        latent_code = self._get_concat_code(model_input)
        reconstructed_model_input = self.decoder(latent_code)
        model_output=Variable(model_input.data+self.delta,requires_grad=False)
        recon_loss_backward = criterion(reconstructed_model_input + self.delta,model_output)
        recon_loss = recon_loss_backward
        recon_loss_backward.backward()
        for opt in opts:
            opt.step()
        self.zero_grad()
        return recon_loss

    def get_category_sample(self,mini_batch_size, num_classes):
        if self.categorical_distribution is None:
            # initialize a distribution to sample from num_classes with equal probability:
            self.categorical_distribution = Categorical(
            probs=numpy.reshape(torch.FloatTensor([1.0 / num_classes] * num_classes * mini_batch_size),
                                    (mini_batch_size, num_classes)))
            self.categories_one_hot = torch.FloatTensor(mini_batch_size, num_classes)

        categories_as_int = self.categorical_distribution.sample().view(mini_batch_size, -1)
        self.categories_one_hot.zero_()
        self.categories_one_hot.scatter_(1,categories_as_int,1)

        categories_real = Variable(self.categories_one_hot.clone(), requires_grad=True)
        return categories_real

    def get_discriminator_loss(self, model_input, *opts):
        self.encoder.eval()
        mini_batch_size = model_input.data.size()[0]

        categories_real=self.get_category_sample(mini_batch_size,num_classes=self.num_classes)
        prior_real = draw_from_gaussian(self.prior_dim, mini_batch_size)

        if model_input.data.is_cuda:
            categories_real=categories_real.cuda()
            prior_real=prior_real.cuda()
        categories_input, prior_input = self.encoder(model_input)
        cat_prob_real = self.discriminator_cat(categories_real)
        cat_prob_input = self.discriminator_cat(categories_input)
        prior_prob_real = self.discriminator_prior(prior_real)
        prior_prob_input = self.discriminator_prior(prior_input)
        cat_discriminator_loss = -torch.mean(
            torch.log(cat_prob_real + self.delta) + torch.log(1 - cat_prob_input + self.delta)
        )
        prior_discriminator_loss = -torch.mean(
            torch.log(prior_prob_real + self.delta) + torch.log(1 - prior_prob_input + self.delta)
        )
        discriminator_loss_backward = cat_discriminator_loss + prior_discriminator_loss
        discriminator_loss = discriminator_loss_backward
        discriminator_loss_backward.backward()
        for opt in opts:
            opt.step()
        self.zero_grad()
        return discriminator_loss

    def get_generator_loss(self, model_input, *opts):
        self.encoder.train()
        categories_input, prior_input = self.encoder(model_input)
        cat_prob_input = self.discriminator_cat(categories_input)
        prior_prob_input = self.discriminator_prior(prior_input)
        generator_loss_backward = -torch.mean(torch.log(cat_prob_input + self.delta))
        generator_loss_backward -= torch.mean(torch.log(prior_prob_input + self.delta))
        generator_loss = generator_loss_backward
        generator_loss_backward.backward()
        for opt in opts:
            opt.step()
        self.zero_grad()
        return generator_loss

    def get_semisup_loss(self, model_input, categories_target, criterion, *opts):
        self.encoder.train()
        categories_input, _ = self.encoder(model_input)
        semisup_loss_backward = criterion(categories_input, categories_target)
        semisup_loss = semisup_loss_backward
        semisup_loss_backward.backward()
        for opt in opts:
            opt.step()
        self.zero_grad()
        return semisup_loss


def create_semisup_adv_autoencoder_model(model_name, problem, encoded_size=32, ngpus=1, nreplicas=1, dropout_p=0,
                                         n_dim=500, num_hidden_layers=1, prior_dim=2):
    input_size = problem.input_size("input")
    num_classes = problem.output_size("softmaxGenotype")
    assert len(input_size) == 1, "AutoEncoders require 1D input features."
    assert len(num_classes) == 1, "AutoEncoders require 1D output features."
    semisup_adv_autoencoder = SemiSupAdvAutoencoder(input_size=input_size[0], n_dim=n_dim, ngpus=ngpus,
                                                    dropout_p=dropout_p, num_hidden_layers=num_hidden_layers,
                                                    num_classes=num_classes[0], prior_dim=prior_dim)
    return semisup_adv_autoencoder