import concurrent
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.backends import cudnn
from torch.nn import MultiLabelSoftMarginLoss, Module

from org.campagnelab.dl.genotypetensors.autoencoder.common_trainer import CommonTrainer, recode_for_label_smoothing
from org.campagnelab.dl.genotypetensors.autoencoder.genotype_softmax_classifier import GenotypeSoftmaxClassifer

from org.campagnelab.dl.genotypetensors.structured.Models import BatchOfInstances, LoadedTensor
from org.campagnelab.dl.genotypetensors.structured.SbiMappers import configure_mappers, BatchOfRecords
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider, DataProvider
from org.campagnelab.dl.performance.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.performance.FloatHelper import FloatHelper
from org.campagnelab.dl.performance.LossHelper import LossHelper
from org.campagnelab.dl.performance.PerformanceList import PerformanceList
from org.campagnelab.dl.utils.utils import progress_bar, normalize_mean_std
from torchfold import torchfold


class FoldExecutor:
    def __init__(self, count_mapper, sample_mapper, record_mapper):
        self.count_mapper = count_mapper
        self.sample_mapper = sample_mapper
        self.record_mapper=record_mapper

    def root_count_map_count(self, count_value):
        return self.count_mapper.map_count.simple_forward(count_value)

    def root_genomic_context_sequence(self, preloaded):
        return self.count_mapper.map_sequence.simple_forward(preloaded)

    def root_from_sequence_sequence(self, preloaded):
        return self.count_mapper.map_sequence.simple_forward(preloaded)

    def root_ref_base_sequence(self, preloaded):
        return self.count_mapper.map_sequence.simple_forward(preloaded)

    def root_count_reduce_count(self, *tensors):
        return self.count_mapper.reduce_count.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_sample_reduce_count(self, *tensors):
        return self.sample_mapper.reduce_counts.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_record_reduce_count(self, *tensors):
        return self.record_mapper.reduce_samples.forward_flat_inputs(torch.cat(tensors, dim=1))

    def root_count_map_gobyGenotypeIndex(self, value):
        return self.count_mapper.map_gobyGenotypeIndex.simple_forward(value)

    def root_count_map_boolean(self, value):
        return self.count_mapper.map_boolean.loaded_forward(value)

    def root_count_map_sequence(self, value):
        return self.count_mapper.map_sequence.simple_forward(value)

    def nwl_map_nwl(self, mapper, numbers, frequencies):
        numbers = mapper.map_number.simple_forward(numbers)
        frequencies = mapper.map_frequency.simple_forward(frequencies)
        batch_size = frequencies.size(0)
        numbers = numbers.view(batch_size, frequencies.size(1), -1)
        concatenated = torch.cat([
            numbers,
            frequencies], dim=2)

        embedding_size=mapper.map_number.embedding_size+mapper.map_frequency.embedding_size
        mapped= mapper.map_sequence.simple_forward(concatenated.view(batch_size,-1,embedding_size))
        return mapped.view(batch_size,embedding_size)

    def root_count_qualityScoresForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_base_qual, numbers, frequencies)

    def root_count_qualityScoresReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_base_qual, numbers, frequencies)

    def root_count_distanceToStartOfRead_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_distance_to, numbers, frequencies)

    def root_count_distanceToEndOfRead_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_distance_to, numbers, frequencies)

    def root_count_readIndicesReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_read_indices, numbers, frequencies)

    def root_count_readIndicesForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_read_indices, numbers, frequencies)


    def root_count_targetAlignedLengths_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_aligned_lengths, numbers, frequencies)

    def root_count_queryAlignedLengths_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_aligned_lengths, numbers, frequencies)

    def root_count_numVariationsInReads_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_num_var, numbers, frequencies)

    def root_count_readMappingQualityForwardStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_mapping_qual, numbers, frequencies)

    def root_count_readMappingQualityReverseStrand_map_nwl(self, numbers, frequencies):
        return self.nwl_map_nwl(self.count_mapper.frequency_list_mapper_mapping_qual, numbers, frequencies)

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
    def __init__(self, args, sbi_mapper, mapped_features_size, output_size, device, use_batching=True):
        super().__init__()
        self.sbi_mapper = sbi_mapper
        self.device = device
        self.use_batching = use_batching
        self.classifier = GenotypeSoftmaxClassifer(num_inputs=mapped_features_size, target_size=output_size[0],
                                                   num_layers=args.num_layers,
                                                   reduction_rate=args.reduction_rate,
                                                   model_capacity=args.model_capacity,
                                                   dropout_p=args.dropout_probability, ngpus=1, use_selu=args.use_selu,
                                                   skip_batch_norm=args.skip_batch_norm)

    def forward(self, sbi_records):
        return self.classifier(self.sbi_mapper.loaded_forward(sbi_records))



class StructGenotypingSupervisedTrainer(CommonTrainer):
    """Train a genotyping model using structured supervised training."""

    def __init__(self, args, problem, device):
        super().__init__(args, problem, device)
        self.criterion_classifier = None
        self.thread_executor = ThreadPoolExecutor(max_workers=args.num_workers) if args.num_workers > 1 \
            else None
        self.is_preloaded = {"training": False, "validation": False}
        self.cache_loaded_records = {"training": [], "validation": []}
        self.fold=None

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
        self.cache_dataset('training',
                           dataset_loader_subset=self.problem.train_loader_subset_range(0, self.args.num_training),
                           length=self.max_training_examples)

        batch_idx = 0
        cpu_device = torch.device('cpu')
        for preloaded_sbi, target_s, metadata in self.cache_loaded_records['training']:
            preloaded_sbi.to(self.device)
            target_s.to(self.device)
            metadata.to(self.device)
            self.fold=torchfold.Fold(self.fold_executor)
            self.train_one_batch(performance_estimators, batch_idx, preloaded_sbi, target_s.tensor(), metadata.tensor())
            preloaded_sbi.to(cpu_device)
            target_s.to(cpu_device)
            metadata.to(cpu_device)
            batch_idx += 1
            if (batch_idx + 1) * self.mini_batch_size > self.max_training_examples:
                break

        return performance_estimators

    def train_one_batch(self, performance_estimators, batch_idx, sbi, target_s, metadata):
        # outputs used to calculate the loss of the supervised model
        # must be done with the model prior to regularization:
        self.net.train()
        indel_weight = self.args.indel_weight_factor
        snp_weight = 1.0
        self.optimizer_training.zero_grad()
        self.net.zero_grad()
        mapped=sbi.mapper.fold(self.fold, "root", sbi)

        features = self.fold.apply(self.fold_executor,[[mapped]])[0]#self.net(sbi)
        output_s=self.net.classifier(features.view(-1)).view(1,-1)
        output_s_p = self.get_p(output_s)
        _, target_index = torch.max(target_s, dim=1)
        supervised_loss = self.criterion_classifier(output_s, target_s)

        batch_weight = self.estimate_batch_weight(metadata, indel_weight=indel_weight,
                                                  snp_weight=snp_weight)

        weighted_supervised_loss = supervised_loss * batch_weight
        optimized_loss = weighted_supervised_loss
        optimized_loss.backward()
        self.optimizer_training.step()
        performance_estimators.set_metric(batch_idx, "supervised_loss", supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "train_accuracy", supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            if batch_idx % 10 ==1:
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

        performance_estimators = self.create_test_performance_estimators()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()

        self.cache_dataset('validation',
                           dataset_loader_subset=self.problem.validation_loader_subset_range(0,
                                                                                             self.args.num_validation),
                           length=self.max_validation_examples)
        batch_idx = 0
        cpu_device = torch.device('cpu')
        self.net.eval()
        for preloaded_sbi, target_s, metadata in self.cache_loaded_records['validation']:
            preloaded_sbi.to(self.device)
            target_s.to(self.device)
            metadata.to(self.device)
            self.fold = torchfold.Fold(self.fold_executor)
            self.test_one_batch(performance_estimators, batch_idx, preloaded_sbi, target_s.tensor(), errors=None)
            preloaded_sbi.to(cpu_device)
            target_s.to(cpu_device)
            metadata.to(cpu_device)


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

        mapped = sbi.mapper.fold(self.fold, "root", sbi)

        features = self.fold.apply(self.fold_executor, [[mapped]])[0]  # self.net(sbi)
        output_s = self.net.classifier(features.view(-1)).view(1, -1)
        output_s_p = self.get_p(output_s)

        supervised_loss = self.criterion_classifier(output_s, target_s)
        self.estimate_errors(errors, output_s, target_s)
        _, target_index = torch.max(target_s, dim=1)
        _, output_index = torch.max(output_s_p, dim=1)
        performance_estimators.set_metric(batch_idx, "test_supervised_loss", supervised_loss.item())
        performance_estimators.set_metric_with_outputs(batch_idx, "test_accuracy", supervised_loss.item(),
                                                       output_s_p, targets=target_index)
        if not self.args.no_progress:
            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         performance_estimators.progress_message(["test_supervised_loss", "test_reconstruction_loss",
                                                                  "test_accuracy"]))

    def create_struct_model(self, problem, args, device=torch.device('cpu')):

        sbi_mappers_configuration = configure_mappers(ploidy=args.struct_ploidy,
                                                      extra_genotypes=args.struct_extra_genotypes,
                                                      num_samples=1, count_dim=args.struct_count_dim,
                                                      sample_dim=args.struct_sample_dim, device=device)
        mappers = sbi_mappers_configuration[0]
        sbi_mapper = BatchOfRecords(mappers['BaseInformation'], device)
        self.fold_executor = FoldExecutor(count_mapper=mappers['CountInfo'], sample_mapper=mappers['SampleInfo'], record_mapper=mappers['BaseInformation'])

        # determine feature size:
        import ujson
        record = ujson.loads(sbi_json_string)
        preloaded = sbi_mapper.preload([record])
        preloaded.to(device)
        mapped = sbi_mapper.loaded_forward(preloaded)
        mapped_features_size = mapped.size(1)

        output_size = problem.output_size("softmaxGenotype")
        model = StructGenotypingModel(args, sbi_mapper, mapped_features_size, output_size, self.device,
                                      args.use_batching)
        print(model)
        return model

    def cache_dataset(self, dataset, dataset_loader_subset, length):
        if not self.is_preloaded[dataset]:
            # Load all records into memory:
            print("Caching {} records into memory..".format(dataset))
            self.cache_loaded_records[dataset]=[]
            data_provider = MultiThreadedCpuGpuDataProvider(
                iterator=zip(dataset_loader_subset),
                device=torch.device('cpu'),
                batch_names=["dataset"],
                requires_grad={"dataset": ["sbi"]},
                recode_functions={
                    "softmaxGenotype": lambda x: recode_for_label_smoothing(x, self.epsilon),
                },
                vectors_to_keep=["metaData"]
            )
            cudnn.benchmark = False
            try:

                for batch_idx, (_, data_dict) in enumerate(data_provider):
                    sbi_batch = data_dict["dataset"]["sbi"]
                    target_s = data_dict["dataset"]["softmaxGenotype"]
                    metadata = data_dict["dataset"]["metaData"]
                    preloaded_sbi_tensors = self.net.sbi_mapper.preload(sbi_batch)
                    for example_index in range(target_s.size(0)):
                        self.cache_loaded_records[dataset].append((preloaded_sbi_tensors[example_index],
                                                                   LoadedTensor(target_s[example_index].view(1, -1)),
                                                                   LoadedTensor(metadata[example_index].view(1, -1))))
                    if not self.args.no_progress:
                        progress_bar(batch_idx * self.mini_batch_size, length, )
            finally:
                data_provider.close()
            self.is_preloaded[dataset]=True
            print("Done caching {}.".format(dataset))


sbi_json_string = '{"type":"BaseInformation","referenceBase":"A","genomicSequenceContext":"GCAGATATACTTCACAGCCCACGCTGACTCTGCCAAGCACA","samples":[{"type":"SampleInfo","counts":[{"type":"CountInfo","matchesReference":true,"isCalled":true,"isIndel":false,"fromSequence":"A","toSequence":"A","genotypeCountForwardStrand":7,"genotypeCountReverseStrand":32,"gobyGenotypeIndex":0,"qualityScoresForwardStrand":[{"type":"NumberWithFrequency","frequency":7,"number":40}],"qualityScoresReverseStrand":[{"type":"NumberWithFrequency","frequency":32,"number":40}],"readIndicesForwardStrand":[{"type":"NumberWithFrequency","frequency":1,"number":23},{"type":"NumberWithFrequency","frequency":1,"number":30},{"type":"NumberWithFrequency","frequency":5,"number":34}],"readIndicesReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":6},{"type":"NumberWithFrequency","frequency":1,"number":22},{"type":"NumberWithFrequency","frequency":1,"number":28},{"type":"NumberWithFrequency","frequency":1,"number":31},{"type":"NumberWithFrequency","frequency":1,"number":34},{"type":"NumberWithFrequency","frequency":1,"number":35},{"type":"NumberWithFrequency","frequency":1,"number":44},{"type":"NumberWithFrequency","frequency":1,"number":50},{"type":"NumberWithFrequency","frequency":1,"number":62},{"type":"NumberWithFrequency","frequency":1,"number":63},{"type":"NumberWithFrequency","frequency":1,"number":68},{"type":"NumberWithFrequency","frequency":2,"number":75},{"type":"NumberWithFrequency","frequency":2,"number":76},{"type":"NumberWithFrequency","frequency":1,"number":81},{"type":"NumberWithFrequency","frequency":1,"number":83},{"type":"NumberWithFrequency","frequency":1,"number":88},{"type":"NumberWithFrequency","frequency":1,"number":89},{"type":"NumberWithFrequency","frequency":1,"number":100},{"type":"NumberWithFrequency","frequency":1,"number":104},{"type":"NumberWithFrequency","frequency":1,"number":109},{"type":"NumberWithFrequency","frequency":1,"number":117},{"type":"NumberWithFrequency","frequency":1,"number":118},{"type":"NumberWithFrequency","frequency":1,"number":125},{"type":"NumberWithFrequency","frequency":1,"number":133},{"type":"NumberWithFrequency","frequency":2,"number":138},{"type":"NumberWithFrequency","frequency":4,"number":139}],"readMappingQualityForwardStrand":[{"type":"NumberWithFrequency","frequency":7,"number":60}],"readMappingQualityReverseStrand":[{"type":"NumberWithFrequency","frequency":32,"number":60}],"numVariationsInReads":[{"type":"NumberWithFrequency","frequency":15,"number":0},{"type":"NumberWithFrequency","frequency":12,"number":1},{"type":"NumberWithFrequency","frequency":8,"number":2},{"type":"NumberWithFrequency","frequency":4,"number":3}],"insertSizes":[{"type":"NumberWithFrequency","frequency":1,"number":-520},{"type":"NumberWithFrequency","frequency":1,"number":-488},{"type":"NumberWithFrequency","frequency":1,"number":-481},{"type":"NumberWithFrequency","frequency":1,"number":-469},{"type":"NumberWithFrequency","frequency":1,"number":-467},{"type":"NumberWithFrequency","frequency":1,"number":-450},{"type":"NumberWithFrequency","frequency":1,"number":-441},{"type":"NumberWithFrequency","frequency":1,"number":-429},{"type":"NumberWithFrequency","frequency":1,"number":-427},{"type":"NumberWithFrequency","frequency":1,"number":-412},{"type":"NumberWithFrequency","frequency":1,"number":-411},{"type":"NumberWithFrequency","frequency":1,"number":-382},{"type":"NumberWithFrequency","frequency":1,"number":-375},{"type":"NumberWithFrequency","frequency":1,"number":-367},{"type":"NumberWithFrequency","frequency":1,"number":-361},{"type":"NumberWithFrequency","frequency":1,"number":-356},{"type":"NumberWithFrequency","frequency":1,"number":-349},{"type":"NumberWithFrequency","frequency":1,"number":-342},{"type":"NumberWithFrequency","frequency":1,"number":-339},{"type":"NumberWithFrequency","frequency":2,"number":-337},{"type":"NumberWithFrequency","frequency":1,"number":-310},{"type":"NumberWithFrequency","frequency":1,"number":-301},{"type":"NumberWithFrequency","frequency":1,"number":-294},{"type":"NumberWithFrequency","frequency":1,"number":-292},{"type":"NumberWithFrequency","frequency":1,"number":-274},{"type":"NumberWithFrequency","frequency":6,"number":0},{"type":"NumberWithFrequency","frequency":1,"number":318},{"type":"NumberWithFrequency","frequency":1,"number":339},{"type":"NumberWithFrequency","frequency":1,"number":397},{"type":"NumberWithFrequency","frequency":1,"number":398},{"type":"NumberWithFrequency","frequency":1,"number":410},{"type":"NumberWithFrequency","frequency":1,"number":426},{"type":"NumberWithFrequency","frequency":1,"number":511}],"targetAlignedLengths":[{"type":"NumberWithFrequency","frequency":2,"number":39},{"type":"NumberWithFrequency","frequency":2,"number":55},{"type":"NumberWithFrequency","frequency":2,"number":61},{"type":"NumberWithFrequency","frequency":2,"number":64},{"type":"NumberWithFrequency","frequency":2,"number":67},{"type":"NumberWithFrequency","frequency":2,"number":68},{"type":"NumberWithFrequency","frequency":2,"number":69},{"type":"NumberWithFrequency","frequency":2,"number":77},{"type":"NumberWithFrequency","frequency":2,"number":82},{"type":"NumberWithFrequency","frequency":4,"number":83},{"type":"NumberWithFrequency","frequency":2,"number":86},{"type":"NumberWithFrequency","frequency":2,"number":95},{"type":"NumberWithFrequency","frequency":2,"number":96},{"type":"NumberWithFrequency","frequency":2,"number":101},{"type":"NumberWithFrequency","frequency":4,"number":108},{"type":"NumberWithFrequency","frequency":4,"number":109},{"type":"NumberWithFrequency","frequency":2,"number":114},{"type":"NumberWithFrequency","frequency":2,"number":116},{"type":"NumberWithFrequency","frequency":4,"number":121},{"type":"NumberWithFrequency","frequency":2,"number":132},{"type":"NumberWithFrequency","frequency":2,"number":136},{"type":"NumberWithFrequency","frequency":2,"number":142},{"type":"NumberWithFrequency","frequency":2,"number":144},{"type":"NumberWithFrequency","frequency":8,"number":150},{"type":"NumberWithFrequency","frequency":4,"number":151},{"type":"NumberWithFrequency","frequency":12,"number":171}],"queryAlignedLengths":[{"type":"NumberWithFrequency","frequency":1,"number":39},{"type":"NumberWithFrequency","frequency":1,"number":55},{"type":"NumberWithFrequency","frequency":1,"number":61},{"type":"NumberWithFrequency","frequency":1,"number":64},{"type":"NumberWithFrequency","frequency":1,"number":67},{"type":"NumberWithFrequency","frequency":1,"number":68},{"type":"NumberWithFrequency","frequency":1,"number":69},{"type":"NumberWithFrequency","frequency":1,"number":77},{"type":"NumberWithFrequency","frequency":1,"number":82},{"type":"NumberWithFrequency","frequency":2,"number":83},{"type":"NumberWithFrequency","frequency":1,"number":86},{"type":"NumberWithFrequency","frequency":1,"number":95},{"type":"NumberWithFrequency","frequency":1,"number":96},{"type":"NumberWithFrequency","frequency":1,"number":101},{"type":"NumberWithFrequency","frequency":2,"number":108},{"type":"NumberWithFrequency","frequency":2,"number":109},{"type":"NumberWithFrequency","frequency":1,"number":114},{"type":"NumberWithFrequency","frequency":1,"number":116},{"type":"NumberWithFrequency","frequency":1,"number":121},{"type":"NumberWithFrequency","frequency":1,"number":122},{"type":"NumberWithFrequency","frequency":1,"number":133},{"type":"NumberWithFrequency","frequency":1,"number":137},{"type":"NumberWithFrequency","frequency":1,"number":142},{"type":"NumberWithFrequency","frequency":1,"number":145},{"type":"NumberWithFrequency","frequency":1,"number":150},{"type":"NumberWithFrequency","frequency":5,"number":151},{"type":"NumberWithFrequency","frequency":2,"number":171},{"type":"NumberWithFrequency","frequency":4,"number":172}],"queryPositions":[{"type":"NumberWithFrequency","frequency":39,"number":0}],"pairFlags":[{"type":"NumberWithFrequency","frequency":6,"number":16},{"type":"NumberWithFrequency","frequency":14,"number":83},{"type":"NumberWithFrequency","frequency":6,"number":99},{"type":"NumberWithFrequency","frequency":12,"number":147},{"type":"NumberWithFrequency","frequency":1,"number":163}],"distancesToReadVariationsForwardStrand":[{"type":"NumberWithFrequency","frequency":2,"number":-70},{"type":"NumberWithFrequency","frequency":4,"number":-29}],"distancesToReadVariationsReverseStrand":[{"type":"NumberWithFrequency","frequency":2,"number":-24},{"type":"NumberWithFrequency","frequency":1,"number":-15},{"type":"NumberWithFrequency","frequency":1,"number":-2},{"type":"NumberWithFrequency","frequency":1,"number":12},{"type":"NumberWithFrequency","frequency":1,"number":13},{"type":"NumberWithFrequency","frequency":1,"number":15},{"type":"NumberWithFrequency","frequency":13,"number":29},{"type":"NumberWithFrequency","frequency":1,"number":49},{"type":"NumberWithFrequency","frequency":3,"number":62},{"type":"NumberWithFrequency","frequency":9,"number":70},{"type":"NumberWithFrequency","frequency":1,"number":73}],"distanceToStartOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":18},{"type":"NumberWithFrequency","frequency":1,"number":23},{"type":"NumberWithFrequency","frequency":1,"number":26},{"type":"NumberWithFrequency","frequency":1,"number":30},{"type":"NumberWithFrequency","frequency":30,"number":33},{"type":"NumberWithFrequency","frequency":5,"number":34}],"distanceToEndOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":6},{"type":"NumberWithFrequency","frequency":1,"number":22},{"type":"NumberWithFrequency","frequency":1,"number":28},{"type":"NumberWithFrequency","frequency":1,"number":31},{"type":"NumberWithFrequency","frequency":1,"number":34},{"type":"NumberWithFrequency","frequency":2,"number":35},{"type":"NumberWithFrequency","frequency":1,"number":44},{"type":"NumberWithFrequency","frequency":1,"number":48},{"type":"NumberWithFrequency","frequency":1,"number":49},{"type":"NumberWithFrequency","frequency":1,"number":50},{"type":"NumberWithFrequency","frequency":1,"number":52},{"type":"NumberWithFrequency","frequency":1,"number":62},{"type":"NumberWithFrequency","frequency":1,"number":63},{"type":"NumberWithFrequency","frequency":1,"number":68},{"type":"NumberWithFrequency","frequency":2,"number":75},{"type":"NumberWithFrequency","frequency":2,"number":76},{"type":"NumberWithFrequency","frequency":1,"number":81},{"type":"NumberWithFrequency","frequency":1,"number":83},{"type":"NumberWithFrequency","frequency":1,"number":88},{"type":"NumberWithFrequency","frequency":1,"number":89},{"type":"NumberWithFrequency","frequency":1,"number":100},{"type":"NumberWithFrequency","frequency":1,"number":104},{"type":"NumberWithFrequency","frequency":1,"number":109},{"type":"NumberWithFrequency","frequency":1,"number":111},{"type":"NumberWithFrequency","frequency":1,"number":117},{"type":"NumberWithFrequency","frequency":1,"number":118},{"type":"NumberWithFrequency","frequency":1,"number":121},{"type":"NumberWithFrequency","frequency":1,"number":125},{"type":"NumberWithFrequency","frequency":1,"number":128},{"type":"NumberWithFrequency","frequency":1,"number":133},{"type":"NumberWithFrequency","frequency":2,"number":138},{"type":"NumberWithFrequency","frequency":4,"number":139}]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"C","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":1,"gobyGenotypeIndex":2,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":7}],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":115}],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":60}],"numVariationsInReads":[{"type":"NumberWithFrequency","frequency":1,"number":2}],"insertSizes":[{"type":"NumberWithFrequency","frequency":1,"number":-301}],"targetAlignedLengths":[{"type":"NumberWithFrequency","frequency":2,"number":148}],"queryAlignedLengths":[{"type":"NumberWithFrequency","frequency":1,"number":148}],"queryPositions":[{"type":"NumberWithFrequency","frequency":1,"number":0}],"pairFlags":[{"type":"NumberWithFrequency","frequency":1,"number":147}],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[{"type":"NumberWithFrequency","frequency":1,"number":-29},{"type":"NumberWithFrequency","frequency":1,"number":0}],"distanceToStartOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":33}],"distanceToEndOfRead":[{"type":"NumberWithFrequency","frequency":1,"number":115}]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"T","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":1,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"G","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":3,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]},{"type":"CountInfo","matchesReference":false,"isCalled":false,"isIndel":false,"fromSequence":"A","toSequence":"N","genotypeCountForwardStrand":0,"genotypeCountReverseStrand":0,"gobyGenotypeIndex":4,"qualityScoresForwardStrand":[],"qualityScoresReverseStrand":[],"readIndicesForwardStrand":[],"readIndicesReverseStrand":[],"readMappingQualityForwardStrand":[],"readMappingQualityReverseStrand":[],"numVariationsInReads":[],"insertSizes":[],"targetAlignedLengths":[],"queryAlignedLengths":[],"queryPositions":[],"pairFlags":[],"distancesToReadVariationsForwardStrand":[],"distancesToReadVariationsReverseStrand":[],"distanceToStartOfRead":[],"distanceToEndOfRead":[]}]}]}'
