from torch import nn


class _SemiSupAdvEncoder(nn.Module):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10, z_dim=2):
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
        self.prior_encoder = nn.Linear(n_dim, z_dim)
    
    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpus > 1:
            cat_encoded = nn.parallel.data_parallel(self.cat_encoder, model_input, self.device_list)
            prior_encoded = nn.parallel.data_parallel(self.prior_encoder, model_input, self.device_list)
        else:
            cat_encoded = self.cat_encoder(model_input)
            prior_encoded = self.prior_encoder(model_input)
        return cat_encoded, prior_encoded


class _SemiSupAdvDecoder(nn.Module):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10, z_dim=2):
        super().__init__()
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        # Create decoder
        decoder_list = [
            nn.BatchNorm1d(z_dim + num_classes),
            nn.Linear(z_dim + num_classes, n_dim),
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
        self.cat_discriminator = nn.Sequential(*self.layer_list)
    
    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpus > 1:
            category = nn.parallel.data_parallel(self.cat_discriminator, model_input, self.device_list)
        else:
            category = self.cat_discriminator(model_input)
        return category


class _SemiSupAdvDiscriminatorPrior(nn.Module):
    def __init__(self, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, z_dim=2):
        super().__init__()
        self.ngpus = ngpus
        self.device_list = list(range(self.ngpus))

        layer_list = [
            nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim, n_dim),
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
        self.prior_discriminator = nn.Sequential(*self.layer_list)
    
    def forward(self, model_input):
        if model_input.data.is_cuda and self.ngpus > 1:
            prior_val = nn.parallel.data_parallel(self.prior_discriminator, model_input, self.device_list)
        else:
            prior_val = self.prior_discriminator(model_input)
        return prior_val


class SemiSupAdvAutoencoder(nn.Module):
    def __init__(self, input_size=512, n_dim=500, ngpus=1, dropout_p=0, num_hidden_layers=3, num_classes=10, z_dim=2):
        super().__init__()
        self.encoder = _SemiSupAdvEncoder(input_size=input_size, n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                          num_hidden_layers=num_hidden_layers, num_classes=num_classes, z_dim=z_dim)
        self.decoder = _SemiSupAdvDecoder(input_size=input_size, n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                          num_hidden_layers=num_hidden_layers, num_classes=num_classes, z_dim=z_dim)
        self.discriminator_cat = _SemiSupAdvDiscriminatorCat(n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                                             num_hidden_layers=num_hidden_layers,
                                                             num_classes=num_classes)
        self.discriminator_prior = _SemiSupAdvDiscriminatorPrior(n_dim=n_dim, ngpus=ngpus, dropout_p=dropout_p,
                                                                 num_hidden_layers=num_hidden_layers,
                                                                 z_dim=z_dim)

    def forward(self, model_input):
        pass


def create_semisup_adv_autoencoder_model(model_name, problem, encoded_size=32, ngpus=1, nreplicas=1, dropout_p=0,
                                         n_dim=500, num_hidden_layers=1, z_dim=2):
    input_size = problem.input_size("input")
    assert len(input_size) == 1, "AutoEncoders required 1D input features."
    semisup_adv_autoencoder = SemiSupAdvAutoencoder(input_size=input_size[0], n_dim=n_dim, ngpus=ngpus,
                                                    dropout_p=dropout_p, num_hidden_layers=num_hidden_layers,
                                                    num_classes=encoded_size, z_dim=z_dim)
    return semisup_adv_autoencoder
