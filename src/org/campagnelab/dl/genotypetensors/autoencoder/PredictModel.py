import sys

from torch.utils.data import DataLoader

from org.campagnelab.dl.genotypetensors.VectorWriterBinary import VectorWriterBinary
from org.campagnelab.dl.genotypetensors.genotype_pytorch_dataset import DispatchDataset
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedDataProvider, DataProvider
from org.campagnelab.dl.problems.StructuredSbiProblem import StructuredSbiGenotypingProblem
from org.campagnelab.dl.utils.utils import progress_bar, normalize_mean_std

from multiprocessing import Lock


class PredictModel:
    def __init__(self, model, device, problem, domain_descriptor=None,
                 feature_mapper=None, samples=None, input_files=None, processing_type="multithreaded",
                 num_workers=0, normalize=False):
        self.model = model
        self.device = device
        self.problem = problem
        self.mini_batch_size = problem.mini_batch_size()
        self.domain_descriptor = domain_descriptor
        self.feature_mapper = feature_mapper
        self.samples = samples
        self.input_files = input_files
        self.processing_type = processing_type
        self.num_workers = num_workers
        self.writer_lock = Lock()
        if normalize:
            problem_mean = problem.load_tensor("input", "mean")
            problem_std = problem.load_tensor("input", "std")
        self.normalize_function = lambda x: normalize_mean_std(x, problem_mean=problem_mean,
                                                               problem_std=problem_std) if normalize \
            else x
        self.input_name = self.problem.get_input_names()[0]
        self.recode_fn = {"input": self.normalize_function} if self.input_name == "input" else None

    def predict(self, iterator, output_filename, max_examples=sys.maxsize):
        self.model.eval()
        if self.processing_type == "multithreaded":
            # Enable fake_GPU_on_CPU to debug on CPU
            data_provider = MultiThreadedDataProvider(iterator=zip(iterator),
                                                      device=self.device,
                                                      batch_names=["unlabeled"],
                                                      recode_functions=self.recode_fn,
                                                      )

        elif self.processing_type == "sequential":
            data_provider = DataProvider(iterator=zip(iterator),
                                         device=self.device,
                                         batch_names=["unlabeled"],
                                         recode_functions=self.recode_fn
                                         )
        else:
            raise Exception("Unrecognized processing type {}".format(self.processing_type))

        with VectorWriterBinary(sample_id=0, path_with_basename=output_filename,
                                tensor_names=self.problem.get_output_names(),
                                domain_descriptor=self.domain_descriptor, feature_mapper=self.feature_mapper,
                                samples=self.samples, input_files=self.input_files, problem=self.problem,
                                model=self.model) as writer:
            for batch_idx, (indices_dict, data_dict) in enumerate(data_provider):
                input_u = data_dict["unlabeled"][self.input_name]
                idxs_u = indices_dict["unlabeled"]
                outputs = self.model(input_u)
                writer.append(list(idxs_u), outputs, inverse_logit=True)
                progress_bar(batch_idx * self.mini_batch_size, max_examples)

                if ((batch_idx + 1) * self.mini_batch_size) > max_examples:
                    break
        data_provider.close()
        print("Done")
