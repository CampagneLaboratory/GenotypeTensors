import numpy
import scipy
import torch
from torch.autograd import Variable
from scipy.stats import norm
from torchnet.meter import ConfusionMeter

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, normalize_mean_std, draw_from_gaussian
from random import *


class AdversarialAutoencoderTrainer(CommonTrainer):
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

    def init_model(self, create_model_function):
        super().init_model(create_model_function)

        self.encoder_semisup_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr,
                                                    weight_decay=self.args.L2)
        self.encoder_generator_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr * 0.8,
                                                      weight_decay=self.args.L2)
        self.encoder_reconstruction_opt = torch.optim.Adam(self.net.encoder.parameters(), lr=self.args.lr * 0.8,
                                                           weight_decay=self.args.L2)
        self.decoder_opt = torch.optim.Adam(self.net.decoder.parameters(), lr=self.args.lr * 0.8,
                                            weight_decay=self.args.L2)
        self.discriminator_prior_opt = torch.optim.Adam(self.net.discriminator_prior.parameters(),
                                                        lr=self.args.lr * 0.6,
                                                        weight_decay=self.args.L2)
        self.discriminator_cat_opt = torch.optim.Adam(self.net.discriminator_cat.parameters(), lr=self.args.lr * 0.6,
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
            performance_estimators += [FloatHelper("semisup_loss")]
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
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        unlabeled_loader = self.problem.unlabeled_loader()

        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset, unlabeled_loader),
            is_cuda=self.use_cuda,
            batch_names=["training", "unlabeled"],
            requires_grad={"training": ["input"], "unlabeled": ["input"]},
            volatile={"training": ["metaData"], "unlabeled": []},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                "input": self.normalize_inputs
            })

        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0

        latent_codes = []
        gaussian_codes = []
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["training"]["input"]
                target_s = data_dict["training"]["softmaxGenotype"]
                input_u = data_dict["unlabeled"]["input"]
                meta_data = data_dict["training"]["metaData"]
                num_batches += 1
                self.zero_grad_all_optimizers()

                self.num_classes = len(target_s[0])
                # Train reconstruction phase:
                self.net.encoder.train()
                self.net.decoder.train()
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
                category_prior = (genotype_frequencies / torch.sum(genotype_frequencies)).numpy()
                discriminator_loss = self.net.get_discriminator_loss(common_trainer=self, model_input=input_u,
                                                                     category_prior=category_prior,
                                                                     recode_labels=lambda x: recode_for_label_smoothing(x,
                                                                                                                        epsilon=self.epsilon))
                discriminator_loss.backward()
                for opt in [self.discriminator_cat_opt, self.discriminator_prior_opt]:
                    opt.step()
                self.zero_grad_all_optimizers()

                # Train generator:
                self.net.encoder.train()
                generator_loss = self.net.get_generator_loss(input_u)
                generator_loss.backward()
                for opt in [self.encoder_generator_opt]:
                    opt.step()
                self.zero_grad_all_optimizers()
                weight = 1
                if self.use_pdf:
                    self.net.encoder.eval()
                    _, latent_code = self.net.encoder(input_s)
                    weight *= self.estimate_example_density_weight(latent_code)

                weight *= self.estimate_batch_weight(meta_data, indel_weight=indel_weight,
                                                     snp_weight=snp_weight)
                self.net.encoder.train()
                semisup_loss = self.net.get_semisup_loss(input_s, target_s) * weight
                semisup_loss.backward()

                for opt in [self.encoder_semisup_opt]:
                    opt.step()
                self.zero_grad_all_optimizers()

                performance_estimators.set_metric(batch_idx, "reconstruction_loss", reconstruction_loss.data[0])
                performance_estimators.set_metric(batch_idx, "discriminator_loss", discriminator_loss.data[0])
                performance_estimators.set_metric(batch_idx, "generator_loss", generator_loss.data[0])
                performance_estimators.set_metric(batch_idx, "semisup_loss", semisup_loss.data[0])
                performance_estimators.set_metric(batch_idx, "weight", weight)

                if self.args.latent_code_output is not None:
                    _, latent_code = self.net.encoder(input_u)
                    # Randomly select n rows from the minibatch to keep track of the latent codes for
                    idxs_to_sample = torch.randperm(latent_code.size()[0])[:self.args.latent_code_n_per_minibatch]
                    for row_idx in idxs_to_sample:
                        latent_code_row = latent_code[row_idx]
                        gaussian_codes.append(torch.squeeze(draw_from_gaussian(latent_code_row.size()[0], 1)))
                        latent_codes.append(latent_code_row)
                if not self.args.no_progress:
                    progress_bar(batch_idx * self.mini_batch_size, self.max_training_examples,
                             performance_estimators.progress_message(
                                 ["reconstruction_loss", "discriminator_loss", "generator_loss", "semisup_loss"]))
                if ((batch_idx + 1) * self.mini_batch_size) > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        if self.args.latent_code_output is not None:
            # Each dimension in latent code should be Gaussian distributed, so take histogram of each column
            # Plot histograms later to see how they compare to Gaussian
            latent_code_tensor = torch.stack(latent_codes).cpu()
            latent_code_histograms = [torch.histc(latent_code_tensor[:, col_idx],
                                                  bins=self.args.latent_code_bins).data.numpy()
                                      for col_idx in range(latent_code_tensor.size()[1])]
            gaussian_code_tensor = torch.stack(gaussian_codes).cpu()
            gaussian_code_histograms = [torch.histc(gaussian_code_tensor[:, col_idx],
                                                    bins=self.args.latent_code_bins).data.numpy()
                                        for col_idx in range(gaussian_code_tensor.size()[1])]
            torch.save({
                "latent": latent_code_histograms,
                "gaussian": gaussian_code_histograms,
            }, "{}_{}.pt".format(self.args.latent_code_output, epoch))
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
        return max(cumulative_pdf, 1 - cumulative_pdf)

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
        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0
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
        cm = ConfusionMeter(self.num_classes, normalized=False)
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["input"]
                target_s = data_dict["validation"]["softmaxGenotype"]
                meta_data = data_dict["validation"]["metaData"]
                # Estimate the reconstruction loss on validation examples:
                reconstruction_loss = self.net.get_reconstruction_loss(input_s)

                # now evaluate prediction of categories:
                categories_predicted, latent_code = self.net.encoder(input_s)
                categories_predicted_p = self.get_p(categories_predicted)
                categories_predicted_p[categories_predicted_p != categories_predicted_p] = 0.0
                _, target_index = torch.max(target_s, dim=1)
                _, output_index = torch.max(categories_predicted_p, dim=1)
                categories_loss = self.net.semisup_loss_criterion(categories_predicted, target_s)
                weight = 1
                if self.use_pdf:

                    weight *= self.estimate_example_density_weight(latent_code)
                else:
                    weight *= self.estimate_batch_weight(meta_data, indel_weight=indel_weight,
                                                         snp_weight=snp_weight)

                cm.add(predicted=output_index.data, target=target_index.data)

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
        self.confusion_matrix = cm.value().transpose()

        if self.best_model_confusion_matrix is None:
            self.best_model_confusion_matrix = torch.from_numpy(self.confusion_matrix)
            if self.use_cuda:
                self.best_model_confusion_matrix = self.best_model_confusion_matrix.cuda()
        return performance_estimators
