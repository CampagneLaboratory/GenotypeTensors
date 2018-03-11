import torch
from torch.autograd import Variable

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar


class AdversarialAutoencoderTrainer(CommonTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_semisup_opt = None
        self.encoder_generator_opt = None
        self.encoder_reconstruction_opt = None
        self.decoder_opt = None
        self.discriminator_prior_opt = None
        self.discriminator_cat_opt = None
        self.optimizers = []

    def init_model(self, create_model_function):
        super().init_model(create_model_function)

        self.encoder_semisup_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr,
                                                   momentum=self.args.momentum, weight_decay=self.args.L2)
        self.encoder_generator_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr,
                                                     momentum=self.args.momentum, weight_decay=self.args.L2)
        self.encoder_reconstruction_opt = torch.optim.SGD(self.net.encoder.parameters(), lr=self.args.lr,
                                                          momentum=self.args.momentum, weight_decay=self.args.L2)
        self.decoder_opt = torch.optim.SGD(self.net.decoder.parameters(), lr=self.args.lr,
                                           momentum=self.args.momentum, weight_decay=self.args.L2)
        self.discriminator_prior_opt = torch.optim.SGD(self.net.discriminator_prior.parameters(), lr=self.args.lr,
                                                       momentum=self.args.momentum, weight_decay=self.args.L2)
        self.discriminator_cat_opt = torch.optim.SGD(self.net.discriminator_cat.parameters(), lr=self.args.lr,
                                                     momentum=self.args.momentum, weight_decay=self.args.L2)
        self.optimizers = [
            self.encoder_semisup_opt,
            self.encoder_generator_opt,
            self.encoder_reconstruction_opt,
            self.decoder_opt,
            self.discriminator_prior_opt,
            self.discriminator_cat_opt,
        ]

    def train_semisup_aae(self, epoch,
                          performance_estimators=None):
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [FloatHelper("reconstruction_loss")]
            performance_estimators += [FloatHelper("discriminator_loss")]
            performance_estimators += [FloatHelper("generator_loss")]
            performance_estimators += [FloatHelper("semisup_loss")]
            print('\nTraining, epoch: %d' % epoch)
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unlabeled_loader = self.problem.unlabeled_loader()
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset, unlabeled_loader),
            is_cuda=self.use_cuda,
            batch_names=["training", "unlabeled"],
            requires_grad={"training": ["input"], "unlabeled": ["input"]},
            volatile={"training": ["metaData"], "unlabeled": []},
            recode_functions={
                "softmaxGenotype": recode_for_label_smoothing
            })

        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0

        for batch_idx, (_, data_dict) in enumerate(data_provider):
            input_s = data_dict["training"]["input"]
            target_s = data_dict["training"]["softmaxGenotype"]
            input_u = data_dict["unlabeled"]["input"]
            meta_data = data_dict["training"]["metaData"]
            num_batches += 1
            self.zero_grad_all_optimizers()

            # Train reconstruction phase:
            #TODO why are we not using the training example as well to train the reconstruction loss?
            self.decoder.train()
            reconstruction_loss = self.net.get_reconstruction_loss(input_u)
            reconstruction_loss.backward()
            for opt in [self.decoder_opt, self.encoder_reconstruction_opt]:
                opt.step()

            # Train discriminators:
            self.net.encoder.eval()
            self.net.discriminator_cat.train()
            self.net.discriminator_prior.train()
            self.zero_grad_all_optimizers()
            genotype_frequencies = self.class_frequencies["softmaxGenotype"]
            category_prior = ( genotype_frequencies/torch.sum(genotype_frequencies)).numpy()
            discriminator_loss = self.net.get_discriminator_loss(input_u, category_prior=category_prior)
            discriminator_loss.backward()
            for opt in [self.discriminator_cat_opt, self.discriminator_prior_opt]:
                opt.step()
            self.zero_grad_all_optimizers()

            # Train generator:
            self.net.encoder.train()
            generator_loss = self.net.get_generator_loss(input_u )
            generator_loss.backward()
            for opt in [self.encoder_generator_opt]:
                opt.step()
            self.zero_grad_all_optimizers()

            self.net.encoder.train()
            semisup_loss = self.net.get_semisup_loss(input_s, target_s) \
                * self.estimate_batch_weight(meta_data, indel_weight=indel_weight,
                                       snp_weight=snp_weight)
            semisup_loss.backward()

            for opt in [self.encoder_semisup_opt]:
                opt.step()
            self.zero_grad_all_optimizers()

            performance_estimators.set_metric(batch_idx, "reconstruction_loss", reconstruction_loss.data[0])
            performance_estimators.set_metric(batch_idx, "discriminator_loss", discriminator_loss.data[0])
            performance_estimators.set_metric(batch_idx, "generator_loss", generator_loss.data[0])
            performance_estimators.set_metric(batch_idx, "semisup_loss", semisup_loss.data[0])

            progress_bar(batch_idx * self.mini_batch_size, self.max_training_examples,
                         performance_estimators.progress_message(["reconstruction_loss","discriminator_loss","generator_loss","semisup_loss"]))
            if ((batch_idx + 1) * self.mini_batch_size) > self.max_training_examples:
                break
        return performance_estimators

    def zero_grad_all_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def test_semisup_aae(self, epoch, performance_estimators=None):
        print('\nTesting, epoch: %d' % epoch)
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [FloatHelper("reconstruction_loss")]
            performance_estimators += [LossHelper("test_loss")]
            performance_estimators += [AccuracyHelper("test_")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset=self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = MultiThreadedCpuGpuDataProvider(iterator=zip(validation_loader_subset),
                                                        is_cuda=self.use_cuda,
                                                        batch_names=["validation"],
                                                        requires_grad={"validation": []},
                                                        volatile={"validation": ["input", "softmaxGenotype"]})

        for batch_idx, (_, data_dict) in enumerate(data_provider):
            input_s = data_dict["validation"]["input"]
            target_s = data_dict["validation"]["softmaxGenotype"]

            # Estimate the reconstruction loss on validation examples:
            reconstruction_loss=self.net.get_reconstruction_loss(input_s)

            # now evaluate prediction of categories:
            categories_predicted, latent_code = self.net.encoder(input_s)
            categories_predicted_p = self.get_p(categories_predicted)
            _, target_index = torch.max(target_s, dim=1)
            categories_loss = self.net.semisup_loss_criterion(categories_predicted_p, target_s)

            performance_estimators.set_metric(batch_idx, "reconstruction_loss", reconstruction_loss.data[0])
            performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", reconstruction_loss.data[0],categories_predicted,target_index)
            performance_estimators.set_metric_with_outputs(batch_idx, "test_loss", categories_loss.data[0], categories_predicted, target_s)

            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_loss","test_accuracy","reconstruction_loss"]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        # print()

        # Apply learning rate schedule:
        test_accuracy = performance_estimators.get_metric("test_loss")
        assert test_accuracy is not None, "test_loss must be found among estimated performance metrics"
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_accuracy, epoch)
        return performance_estimators
