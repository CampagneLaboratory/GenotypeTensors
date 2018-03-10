import torch
from torch import nn

from org.campagnelab.dl.utils.utils import draw_from_categorical, draw_from_gaussian


class _SemiSupAdvEncoder(nn.Module):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10,
                 prior_dim=2):
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
        self.cat_encoder = nn.Linear(n_dim, num_classes)
        self.prior_encoder = nn.Linear(n_dim, prior_dim)
    
    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpus > 1:
            cat_encoded = nn.parallel.data_parallel(self.cat_encoder, model_input, self.device_list)
            prior_encoded = nn.parallel.data_parallel(self.prior_encoder, model_input, self.device_list)
        else:
            cat_encoded = self.cat_encoder(model_input)
            prior_encoded = self.prior_encoder(model_input)
        return cat_encoded, prior_encoded


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
    
    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpus > 1:
            category = nn.parallel.data_parallel(self.cat_discriminator, model_input, self.device_list)
        else:
            category = self.cat_discriminator(model_input)
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
                 delta=1E-15, seed=None):
        super().__init__()
        self.encoder = _SemiSupAdvEncoder(input_size=input_size, n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                          num_hidden_layers=num_hidden_layers, num_classes=num_classes,
                                          prior_dim=prior_dim)
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

    def forward(self, model_input):
        return self.decoder(self._get_concat_code(model_input))

    def _get_concat_code(self, model_input):
        latent_categorical_code, latent_gaussian_code = self.encoder(model_input)
        return torch.cat(latent_categorical_code, latent_gaussian_code, 1)

    def get_reconstruction_loss(self, model_input, criterion, *opts):
        self.decoder.train()
        latent_code = self._get_concat_code(model_input)
        reconstructed_model_input = self.decoder(latent_code)
        recon_loss_backward = criterion(model_input + self.delta, reconstructed_model_input + self.delta)
        recon_loss = recon_loss_backward
        recon_loss_backward.backward()
        for opt in opts:
            opt.step()
        self.zero_grad()
        return recon_loss

    def get_discriminator_loss(self, model_input, *opts):
        self.encoder.eval()
        mini_batch_size = model_input.data.size()[0]
        categories_real = draw_from_categorical(self.num_classes, mini_batch_size)
        prior_real = draw_from_gaussian(self.prior_dim, mini_batch_size)
        if model_input.data.is_cuda:
            categories_real.cuda()
            prior_real.cuda()
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
    assert len(input_size) == 1, "AutoEncoders required 1D input features."
    semisup_adv_autoencoder = SemiSupAdvAutoencoder(input_size=input_size[0], n_dim=n_dim, ngpus=ngpus,
                                                    dropout_p=dropout_p, num_hidden_layers=num_hidden_layers,
                                                    num_classes=encoded_size, prior_dim=prior_dim)
    return semisup_adv_autoencoder
