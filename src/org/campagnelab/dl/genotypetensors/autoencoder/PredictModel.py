import sys

from org.campagnelab.dl.genotypetensors.VectorWriterBinary import VectorWriterBinary
from org.campagnelab.dl.multithreading.sequential_implementation import MultiThreadedCpuGpuDataProvider
from org.campagnelab.dl.utils.utils import progress_bar


class PredictModel:
    def __init__(self, model, use_cuda, problem):
        self.model = model
        self.use_cuda = use_cuda
        self.problem = problem
        self.mini_batch_size = problem.mini_batch_size()

    def predict(self, iterator, output_filename, max_examples=sys.maxsize):

        self.model.eval()
        data_provider = MultiThreadedCpuGpuDataProvider(iterator=zip(iterator),
                                                        is_cuda=self.use_cuda,
                                                        batch_names=["unlabeled"],
                                                        volatile={"unlabeled": ["input"]})

        with VectorWriterBinary(sample_id=0, path_with_basename=output_filename,
                                tensor_names=self.problem.get_output_names()) as writer:
            for batch_idx, dict in enumerate(data_provider):
                input_u = dict["unlabeled"]["input"]

                outputs = self.model(input_u)
                writer.append(0, outputs)
                progress_bar(batch_idx * self.mini_batch_size, max_examples)

                if ((batch_idx + 1) * self.mini_batch_size) > max_examples:
                    break

        data_provider.close()
        print("Done")