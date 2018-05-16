import concurrent
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.backends import cudnn
from torch.nn import MultiLabelSoftMarginLoss, Module

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.genotypetensors.autoencoder.genotype_softmax_classifier import GenotypeSoftmaxClassifer
from org.campagnelab.dl.genotypetensors.structured.BatchedSbiMappers import configure_mappers, sbi_json_string

from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider, DataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, normalize_mean_std


def chunks(l, n):
    """Yield successive n-sized chunks from list l.
    example:
    list(chunks([1,2,3,4,5,6,7],2)) returns:            [[1, 2], [3, 4], [5, 6], [7]]
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def to_binary(n, max_value):
    for index in list(range(max_value))[::-1]:
        yield 1 & int(n) >> index


enable_recode = False


class StructGenotypingModel(Module):
    def __init__(self, args, sbi_mapper, mapped_features_size, output_size, use_cuda, use_batching=True):
        super().__init__()
        self.sbi_mapper = sbi_mapper
        self.use_cuda = use_cuda
        self.use_batching = use_batching
        self.classifier = GenotypeSoftmaxClassifer(num_inputs=mapped_features_size, target_size=output_size[0],
                                                   num_layers=args.num_layers,
                                                   reduction_rate=args.reduction_rate,
                                                   model_capacity=args.model_capacity,
                                                   dropout_p=args.dropout_probability, ngpus=1, use_selu=args.use_selu,
                                                   skip_batch_norm=args.skip_batch_norm)

    def map_sbi_messages(self, sbi_records):


        if self.use_batching:
            tensors=self.sbi_mapper.collect_tensors(sbi_records, "records",{})
            features=self.sbi_mapper.forward_batch(sbi_records, "records", tensors,
                                                   index_maps=[record['indices'] for record in sbi_records])
        else:
            # Create a new cache for each mini-batch because we cache embeddings:

            features = self.sbi_mapper(sbi_records)
        return features

    def forward(self, sbi_records):
        return self.classifier(self.map_sbi_messages(sbi_records))


class StructGenotypingSupervisedTrainer(CommonTrainer):
    """Train a genotyping model using structured supervised training."""

    def __init__(self, args, problem, use_cuda):
        super().__init__(args, problem, use_cuda)
        self.criterion_classifier = None
        self.thread_executor = ThreadPoolExecutor(max_workers=args.num_workers) if args.num_workers > 1 \
            else None
        self.tensor_cache = None

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
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(train_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["training"],
            requires_grad={"training": ["sbi"]},
            volatile={"training": ["metaData"]},
            recode_functions={
                "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
            }
        )
        cudnn.benchmark = False
        try:

            for batch_idx, (_, data_dict) in enumerate(data_provider):
                sbi = data_dict["training"]["sbi"]
                target_s = data_dict["training"]["softmaxGenotype"]
                metadata = data_dict["training"]["metaData"]

                self.train_one_batch(performance_estimators, batch_idx, sbi, target_s, metadata)
                if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                    break
        finally:
            data_provider.close()

        return performance_estimators

    def train_one_batch(self, performance_estimators, batch_idx, sbi, target_s, metadata):
        # outputs used to calculate the loss of the supervised model
        # must be done with the model prior to regularization:
        self.net.train()
        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0
        self.optimizer_training.zero_grad()
        self.net.zero_grad()

        output_s = self.net(sbi)
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

    def map_sbi(self, sbi):
        # process mapping of sbi messages in parallel:
        if self.thread_executor is not None:
            def todo(net, records, tensor_cache, cuda):
                # print("processing batch")
                features = net.map_sbi_messages(records, tensor_cache=tensor_cache, cuda=cuda)
                return features

            futures = []
            records_per_worker = self.args.mini_batch_size // self.args.num_workers
            for record in chunks(sbi, records_per_worker):
                futures += [self.thread_executor.submit(todo, self.net, record,
                                                        tensor_cache=self.tensor_cache, cuda=self.use_cuda)]
            concurrent.futures.wait(futures)
            input_s = torch.cat([future.result() for future in futures], dim=0)
        else:
            input_s = self.net.map_sbi_messages(sbi, self.tensor_cache, cuda=self.use_cuda)
        return input_s

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
        data_provider = MultiThreadedCpuGpuDataProvider(
            iterator=zip(validation_loader_subset),
            is_cuda=self.use_cuda,
            batch_names=["validation"],
            requires_grad={"validation": []},
            volatile={
                "validation": ["sbi", "softmaxGenotype"]
            },

        )
        try:
            for batch_idx, (_, data_dict) in enumerate(data_provider):
                sbi = data_dict["validation"]["sbi"]
                target_s = data_dict["validation"]["softmaxGenotype"]
                self.net.eval()
                self.test_one_batch(performance_estimators, batch_idx, sbi, target_s, errors=errors)

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

    def test_one_batch(self, performance_estimators, batch_idx, sbi, target_s, metadata=None, errors=None):
        if errors is None:
            errors = torch.zeros(target_s[0].size())

        output_s = self.net(sbi)
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

    def create_struct_model(self, problem, args,use_cuda):

        sbi_mappers, *_ = configure_mappers(ploidy=args.struct_ploidy,
                                                      extra_genotypes=args.struct_extra_genotypes,
                                                      num_samples=1, count_dim=args.struct_count_dim,
                                                      sample_dim=args.struct_sample_dim,use_cuda=use_cuda)
        sbi_mapper = sbi_mappers['BaseInformation']
        # determine feature size:


        import ujson
        record = ujson.loads(sbi_json_string)
        if self.use_cuda:
            sbi_mapper.cuda()
        mapped_features_size = sbi_mapper(record).size(1)

        output_size = problem.output_size("softmaxGenotype")
        model = StructGenotypingModel(args, sbi_mapper, mapped_features_size, output_size, self.use_cuda,
                                      args.use_batching)
        print(model)
        return model
