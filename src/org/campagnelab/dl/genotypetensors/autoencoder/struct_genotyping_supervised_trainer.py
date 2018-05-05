import torch
from torch.nn import MultiLabelSoftMarginLoss, Module

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.genotypetensors.autoencoder.genotype_softmax_classifier import GenotypeSoftmaxClassifer
from org.campagnelab.dl.genotypetensors.structured.Models import BatchOfInstances
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import configure_mappers
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider, DataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, normalize_mean_std


def to_binary(n, max_value):
    for index in list(range(max_value))[::-1]:
        yield 1 & int(n) >> index


enable_recode = False


class StructGenotypingModel(Module):
    def __init__(self, args, sbi_mapper, mapped_features_size, output_size):
        super().__init__()
        self.sbi_mapper = sbi_mapper
        self.classifier = GenotypeSoftmaxClassifer(num_inputs=mapped_features_size, target_size=output_size[0],
                                                   num_layers=args.num_layers,
                                                   reduction_rate=args.reduction_rate,
                                                   model_capacity=args.model_capacity,
                                                   dropout_p=args.dropout_probability, ngpus=1, use_selu=args.use_selu,
                                                   skip_batch_norm=args.skip_batch_norm)

    def map_sbi_messages(self,sbi_records):
        features = self.sbi_mapper(sbi_records)
        return features

    def forward(self, mapped_features):

        return self.classifier(mapped_features)


class StructGenotypingSupervisedTrainer(CommonTrainer):
    """Train a genotyping model using structured supervised training."""

    def __init__(self, args, problem, use_cuda):
        super().__init__(args, problem, use_cuda)
        self.criterion_classifier = None

    def rebuild_criterions(self, output_name, weights=None):
        if output_name == "softmaxGenotype":
            self.criterion_classifier = MultiLabelSoftMarginLoss(weight=weights)

    def get_test_metric_name(self):
        return "test_accuracy"

    def is_better(self, metric, previous_metric):
        return metric > previous_metric

    def train_supervised(self, epoch):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("supervised_loss")]
        performance_estimators += [AccuracyHelper("train_")]

        print('\nTraining, epoch: %d' % epoch)

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        unsupervised_loss_acc = 0
        num_batches = 0
        train_loader_subset = self.problem.train_loader_subset_range(0, self.args.num_training)
        data_provider = DataProvider(
            iterator=zip(train_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["training"],
            requires_grad={"training": ["sbi"]},
            volatile={"training": ["metaData"]},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                "sbi": lambda messages: self.net.map_sbi_messages(messages)
            }
        )
        try:

            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["training"]["sbi"]
                target_s = data_dict["training"]["softmaxGenotype"]
                metadata = data_dict["training"]["metaData"]

                self.train_one_batch(performance_estimators, batch_idx, input_s, target_s, metadata)

                if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        return performance_estimators

    def train_one_batch(self, performance_estimators, batch_idx, input_s, target_s, metadata):
        # outputs used to calculate the loss of the supervised model
        # must be done with the model prior to regularization:
        self.net.train()
        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0
        self.optimizer_training.zero_grad()
        self.net.zero_grad()
        output_s = self.net(input_s)
        output_s_p = self.get_p(output_s)
        _, target_index = torch.max(target_s, dim=1)
        supervised_loss = self.criterion_classifier(output_s, target_s)

        batch_weight = self.estimate_batch_weight(metadata, indel_weight=indel_weight,
                                                  snp_weight=snp_weight)

        weighted_supervised_loss = supervised_loss * batch_weight
        optimized_loss = weighted_supervised_loss
        optimized_loss.backward()
        self.optimizer_training.step()
        performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.data[0])
        performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.data[0],
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size,
                         self.max_training_examples,
                         performance_estimators.progress_message(
                             ["supervised_loss", "reconstruction_loss", "train_accuracy"]))

    def get_p(self, output_s):
        # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
        # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
        output_s_exp = torch.exp(output_s)
        output_s_p = torch.div(output_s_exp, torch.add(output_s_exp, 1))
        return output_s_p

    def test_supervised(self, epoch):
        print('\nTesting, epoch: %d' % epoch)
        errors = None
        performance_estimators = self.create_test_performance_estimators()

        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        validation_loader_subset = self.problem.validation_loader_range(0, self.args.num_validation)
        data_provider = DataProvider(
            iterator=zip(validation_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["validation"],
            requires_grad={"validation": []},
            volatile={
                "validation": ["sbi", "softmaxGenotype"]
            },
            recode_functions={
                "sbi": lambda messages: self.net.map_sbi_messages(messages)
            }
        )
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                input_s = data_dict["validation"]["sbi"]
                target_s = data_dict["validation"]["softmaxGenotype"]
                self.net.eval()
                self.test_one_batch(performance_estimators, batch_idx, input_s, target_s, errors=errors)

                if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                    break
            # print()
        finally:
            data_provider.close()
        print("test errors by class: ", str(errors))
        if self.reweight_by_validation_error:
            self.reweight_by_val_errors(errors)
        # Apply learning rate schedule:
        test_metric = performance_estimators.get_metric(self.get_test_metric_name())
        assert test_metric is not None, (self.get_test_metric_name() +
                                         "must be found among estimated performance metrics")
        if not self.args.constant_learning_rates:
            self.scheduler_train.step(test_metric, epoch)
        return performance_estimators

    def create_training_performance_estimators(self):
        performance_estimators = PerformanceList()
        performance_estimators += [FloatHelper("supervised_loss")]
        performance_estimators += [AccuracyHelper("train_")]
        self.training_performance_estimators = performance_estimators
        return performance_estimators

    def create_test_performance_estimators(self):
        performance_estimators = PerformanceList()
        performance_estimators += [LossHelper("test_supervised_loss")]
        performance_estimators += [AccuracyHelper("test_")]
        self.test_performance_estimators = performance_estimators
        return performance_estimators

    def test_one_batch(self, performance_estimators, batch_idx, input_s, target_s, metadata=None, errors=None):
        if errors is None:
            errors = torch.zeros(target_s[0].size())

        output_s = self.net(input_s)
        output_s_p = self.get_p(output_s)

        supervised_loss = self.criterion_classifier(output_s, target_s)
        self.estimate_errors(errors, output_s, target_s)
        _, target_index = torch.max(target_s, dim=1)
        _, output_index = torch.max(output_s_p, dim=1)
        performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.data[0])
        performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", supervised_loss.data[0],
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                  "test_accuracy"]))

    def create_struct_model(self, problem, args):

        sbi_mappers_configuration = configure_mappers(ploidy=2, extra_genotypes=3, num_samples=1)
        sbi_mapper = BatchOfInstances(mappers=sbi_mappers_configuration)
        # determine feature size:

        json_string = '{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4}]}]}'
        import ujson
        record = ujson.loads(json_string)

        mapped_features_size =sbi_mapper([record]).size(1)

        output_size=problem.output_size("softmaxGenotype")
        return StructGenotypingModel(args, sbi_mapper, mapped_features_size, output_size)