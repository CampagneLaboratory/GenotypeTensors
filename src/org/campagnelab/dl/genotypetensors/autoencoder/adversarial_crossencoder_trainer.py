import numpy
import scipy
import torch
from torch.autograd import Variable
from scipy.stats import norm

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, normalize_mean_std
from random import *


class AdversarialCrossencoderTrainer(CommonTrainer):
    def __init__(self, args, problem, use_cuda=False):
        super().__init__(args, problem, use_cuda)
        self.encoder_semisup_opt = None
        self.encoder_generator_opt = None
        self.encoder_reconstruction_opt = None
        self.decoder_opt = None
        self.discriminator_prior_opt = None
        self.discriminator_cat_opt = None
        self.optimizers = []
        self.use_pdf = args.use_density_weights
        self.normalize_inputs = None
        self.schedulers = None

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, previous_metric):
        return metric > previous_metric

    def init_model(self, create_model_function,class_frequencies=None):
        super().init_model(create_model_function,class_frequencies=class_frequencies)

        self.encoder_semisup_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr*0.5,
                                                    weight_decay=self.args.L2)
        self.encoder_generator_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr ,
                                                      weight_decay=self.args.L2)
        self.encoder_reconstruction_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr,
                                                           weight_decay=self.args.L2)
        self.decoder_opt = torch.optim.Adam(self.net.decoder.parameters(), lr=self.args.lr,
                                            weight_decay=self.args.L2)
        self.discriminator_prior_opt = torch.optim.Adam(self.net.discriminator_prior.parameters(),
                                                        lr=self.args.lr * 0.8,
                                                        weight_decay=self.args.L2)
        self.discriminator_cat_opt = torch.optim.Adam(self.net.discriminator_cat.parameters(), lr=self.args.lr * 0.8,
                                                      weight_decay=self.args.L2)

        self.optimizers = [
            self.encoder_semisup_opt,
            self.encoder_generator_opt,
            self.encoder_reconstruction_opt,
            self.decoder_opt,
            self.discriminator_prior_opt,
            self.discriminator_cat_opt,
        ]
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers += [self.create_scheduler_for_optimizer(optimizer)]

        if self.args.normalize:
            problem_mean = self.problem.load_tensor("input", "mean")
            problem_std = self.problem.load_tensor("input", "std")

        self.normalize_inputs = lambda x: normalize_mean_std(x, problem_mean=problem_mean,
                                                             problem_std=problem_std) if self.args.normalize else x

    def train_semisup_aae(self, epoch,
                          performance_estimators=None):
        if performance_estimators is None:
            performance_estimators = PerformanceList()
            performance_estimators += [FloatHelper("reconstruction_loss")]
            performance_estimators += [FloatHelper("discriminator_loss")]
            performance_estimators += [FloatHelper("generator_loss")]
            performance_estimators += [FloatHelper("supervised_loss")]
            performance_estimators += [FloatHelper("weight")]
            print('\nTraining, epoch: %d' % epoch)
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        self.net.train()
        supervised_grad_norm = 1.
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset1 = self.problem.train_loader_subset_range(0, self.args.num_training)
        train_loader_subset2 = self.problem.train_loader_subset_range(0, self.args.num_training)

        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset1, train_loader_subset2),
            is_cuda=self.use_cuda,
            batch_names=["training1", "training2"],
            requires_grad={"training1": ["input"], "training2": ["input"]},
            volatile={"training1": ["metaData"], "training2": ["metaData"]},
            recode_functions={
                "softmaxGenotype": recode_for_label_smoothing,
                "input": self.normalize_inputs
            })

        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0

        latent_codes = []
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s1 = data_dict["training1"]["input"]
                input_s2 = data_dict["training2"]["input"]
                target_s1 = data_dict["training1"]["softmaxGenotype"]
                target_s2 = data_dict["training2"]["softmaxGenotype"]

                meta_data1 = data_dict["training1"]["metaData"]
                meta_data2 = data_dict["training2"]["metaData"]
                num_batches += 1
                self.zero_grad_all_optimizers()

                # input_s=normalize_mean_std(input_s)
                # input_u=normalize_mean_std(input_u)
                # print(torch.mean(input_s,dim=0))
                # Train reconstruction phase:
                self.net.decoder.train()
                reconstruction_loss = self.net.get_crossconstruction_loss(input_s1,input_s2,target_s2)
                reconstruction_loss.backward()
                for opt in [self.decoder_opt, self.encoder_reconstruction_opt]:
                    opt.step()

                # Train discriminators:
                self.net.encoder.eval()
                self.net.discriminator_cat.train()
                self.net.discriminator_prior.train()
                self.zero_grad_all_optimizers()
                genotype_frequencies = self.class_frequencies["softmaxGenotype"]
                category_prior = (genotype_frequencies / torch.sum(genotype_frequencies)).numpy()
                discriminator_loss = self.net.get_discriminator_loss(input_s1, category_prior=category_prior)
                discriminator_loss.backward()
                for opt in [self.discriminator_cat_opt, self.discriminator_prior_opt]:
                    opt.step()
                self.zero_grad_all_optimizers()

                # Train generator:
                self.net.encoder.train()
                generator_loss = self.net.get_generator_loss(input_s1)
                generator_loss.backward()
                for opt in [self.encoder_generator_opt]:
                    opt.step()
                self.zero_grad_all_optimizers()

                if self.use_pdf:
                    self.net.encoder.eval()
                    _, latent_code = self.net.encoder(input_s1)
                    weight = self.estimate_example_density_weight(latent_code)
                else:
                    weight = self.estimate_batch_weight(meta_data1, indel_weight=indel_weight,
                                                        snp_weight=snp_weight)
                self.net.encoder.train()
                supervised_loss = self.net.get_crossencoder_supervised_loss(input_s1, target_s1) * weight
                supervised_loss.backward()

                for opt in [self.encoder_semisup_opt]:
                    opt.step()
                self.zero_grad_all_optimizers()

                performance_estimators.set_metric(batch_idx, "reconstruction_loss", reconstruction_loss.data[0])
                performance_estimators.set_metric(batch_idx, "discriminator_loss", discriminator_loss.data[0])
                performance_estimators.set_metric(batch_idx, "generator_loss", generator_loss.data[0])
                performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.data[0])
                performance_estimators.set_metric(batch_idx, "weight", weight)
                if not self.args.no_progress:
                    progress_bar(batch_idx * self.mini_batch_size, self.max_training_examples,
                             performance_estimators.progress_message(
                                 ["reconstruction_loss", "discriminator_loss", "generator_loss", "semisup_loss"]))
                if ((batch_idx + 1) * self.mini_batch_size) > self.max_training_examples:
                    break
        finally:
            data_provider.close()


        return performance_estimators

    def estimate_example_density_weight(self, latent_code):
        cumulative_pdf = 0
        n_pdf = 0
        for z in latent_code:
            pdf = norm.pdf(z.data)
            # in the early stages of training, pdf larger than 1 if latent variable far from normally distributed
            valid_pdf = pdf[pdf <= 1]
            cumulative_pdf += numpy.sum(valid_pdf)
            n_pdf = n_pdf + len(valid_pdf)
        cumulative_pdf /= n_pdf
        return max(cumulative_pdf, 1-cumulative_pdf)

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
            performance_estimators += [FloatHelper("weight")]

        self.net.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset = self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = MultiThreadedCpuGpuDataProvider(iterator=zip(validation_loader_subset),
                                                        is_cuda=self.use_cuda,
                                                        batch_names=["validation"],
                                                        requires_grad={"validation": []},
                                                        volatile={"validation": ["input", "softmaxGenotype"],
                                                                  },
                                                        recode_functions={
                                                            "input": self.normalize_inputs
                                                        })
        self.net.eval()
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["input"]
                target_s = data_dict["validation"]["softmaxGenotype"]

                # Estimate the reconstruction loss on validation examples:
                reconstruction_loss = self.net.get_crossconstruction_loss(input_s,input_s,target_s)

                # now evaluate prediction of categories:
                categories_predicted, latent_code = self.net.encoder(input_s)
    #            categories_predicted+=self.net.latent_to_categories(latent_code)

                categories_predicted_p = self.get_p(categories_predicted)
                categories_predicted_p[categories_predicted_p != categories_predicted_p] = 0.0
                _, target_index = torch.max(target_s, dim=1)
                categories_loss = self.net.semisup_loss_criterion(categories_predicted, target_s)

                weight = self.estimate_example_density_weight(latent_code)
                performance_estimators.set_metric(batch_idx, "reconstruction_loss", reconstruction_loss.data[0])
                performance_estimators.set_metric(batch_idx, "weight", weight)
                performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", reconstruction_loss.data[0],
                                                               categories_predicted_p, target_index)
                performance_estimators.set_metric_with_outputs(batch_idx, "test_loss", categories_loss.data[0] * weight,
                                                               categories_predicted_p, target_s)

                if not self.args.no_progress:
                    progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                             performance_estimators.progress_message(["test_loss", "test_accuracy", "reconstruction_loss"]))

                if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                    break
            # print()
        finally:
            data_provider.close()
        # Apply learning rate schedules:
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, (self.get_test_metric_name()
                                         + "must be found among estimated performance metrics")
        if not self.args.constant_learning_rates:
            for scheduler in self.schedulers:
                scheduler.step(test_metric, epoch)
        # Run the garbage collector to try to release memory we no longer need:
        import gc
        gc.collect()
        return performance_estimators
