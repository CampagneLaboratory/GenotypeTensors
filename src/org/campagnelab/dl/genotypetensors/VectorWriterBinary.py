import copy
import json
import os
import struct
import warnings

from functools import reduce
from operator import mul

from torch.autograd import Variable

from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader

import torch
import numpy as np

from org.campagnelab.dl.problems.StructuredSbiProblem import StructuredSbiGenotypingProblem


class VectorWriterBinary:
    major_version = 0
    minor_version = 4

    example_id_size = 8
    vector_id_size = 4
    sample_id_size = 4
    vector_length_size = 4
    header_size = example_id_size + vector_id_size + sample_id_size + vector_length_size
    vector_element_size = 4

    def __init__(self, path_with_basename, sample_id, tensor_names, input_data_path=None, domain_descriptor=None,
                 problem=None, feature_mapper=None, samples=None, input_files=None, model=None):
        self.basename = path_with_basename
        self.vec_file = open(self.basename + ".vec", "wb")
        self.sample_id = sample_id
        self.vector_names = tensor_names
        self.output_properties = {}
        self.using_input_data = input_data_path is not None
        self.problem = problem
        self.using_structured_problem = isinstance(self.problem, StructuredSbiGenotypingProblem)
        if self.using_structured_problem:
            assert model is not None, "Need to have model present for use with structured problem"
            base_info = model.sbi_mapper.sbi_mapper

            softmax_genotype_dimension = (2 ** (base_info.ploidy + base_info.extra_genotypes)) + 1
            num_bytes_per_vector_elements = softmax_genotype_dimension * VectorWriterBinary.vector_element_size
            num_bytes_per_vector = num_bytes_per_vector_elements + VectorWriterBinary.header_size
            num_bytes_per_example = num_bytes_per_vector
            self.output_properties = {
                "majorVersion": VectorWriterBinary.major_version,
                "minorVersion": VectorWriterBinary.minor_version,
                "fileType": "binary",
                "domainDescriptor":
                    domain_descriptor if domain_descriptor is not None else "structured_pytorch_model_only",
                "featureMapper": feature_mapper if feature_mapper is not None else "structured_pytorch_model_only",
                "headerSize": VectorWriterBinary.header_size,
                "inputFiles": input_files if input_files is not None else ["structured_pytorch_model_only"],
                "samples": samples if samples is not None else [
                    {
                        "sampleName": "structured_pytorch_model_only",
                        "sampleType": "structured_pytorch_model_only",
                    },
                ],
                "vectors": [
                    {
                        "vectorName": "softmaxGenotype",
                        "vectorType": "float32",
                        "vectorElementSize": VectorWriterBinary.vector_element_size,
                        "vectorDimension": [
                            softmax_genotype_dimension,
                        ],
                        "vectorNumBytesForElements":
                            VectorWriterBinary.vector_element_size * softmax_genotype_dimension
                    },
                ],
                "numBytesPerExample": num_bytes_per_example
            }
            self.num_bytes_per_example = num_bytes_per_example
            self.vector_props_written = True
            self.device = torch.device("cpu")

        elif self.using_input_data:
            def _get_vector_length_from_props(vector_name_for_dims):
                return VectorWriterBinary._get_vector_length(self.input_vector_properties_reader
                                                             .get_vector_dimensions_from_name(vector_name_for_dims))
            self.input_vector_properties_reader = VectorPropertiesReader(
                "{}.vecp".format(input_data_path.split(os.extsep)[0])
            )
            input_properties = self.input_vector_properties_reader.vector_properties
            self.vector_ids = set(self.input_vector_properties_reader.get_vector_idx_from_name(vector_name)
                                  for vector_name in tensor_names)
            self.output_properties = {
                "majorVersion": input_properties["majorVersion"],
                "minorVersion": input_properties["minorVersion"],
                "fileType": "binary",
                "domainDescriptor": input_properties["domainDescriptor"],
                "featureMapper": input_properties["featureMapper"],
                "headerSize": input_properties["headerSize"],
                "inputFiles": copy.deepcopy(input_properties["inputFiles"]),
                "samples": [
                    dict(copy.deepcopy(input_properties["samples"][self.sample_id]))
                ],
                "vectors": [
                    dict(copy.deepcopy(input_properties["vector"][vector_id])) for vector_id in self.vector_ids
                ],
            }
            num_bytes_per_example = (vector["vectorNumBytesPerElements"]
                                     for vector in self.output_properties["vectors"])
            num_bytes_per_example = (num_bytes_per_vector + self.output_properties["headerSize"]
                                     for num_bytes_per_vector in num_bytes_per_example)
            num_bytes_per_example = sum((num_bytes_per_vector * len(self.output_properties["samples"])
                                         for num_bytes_per_vector in num_bytes_per_example))
            self.output_properties["numBytesPerExample"] = num_bytes_per_example
            self.vector_lengths = {vector_name: _get_vector_length_from_props(vector_name)
                                   for vector_name in tensor_names}
            self.num_bytes_per_example = num_bytes_per_example
            self.vector_props_written = True
        else:
            self.output_properties = {
                "majorVersion": VectorWriterBinary.major_version,
                "minorVersion": VectorWriterBinary.minor_version,
                "fileType": "binary",
                "domainDescriptor": domain_descriptor if domain_descriptor is not None else "pytorch_model_only",
                "featureMapper": feature_mapper if feature_mapper is not None else "pytorch_model_only",
                "headerSize": VectorWriterBinary.header_size,
                "inputFiles": input_files if input_files is not None else ["pytorch_model_only"],
                "samples": samples if samples is not None else [
                    {
                        "sampleName": "pytorch_model_only",
                        "sampleType": "pytorch_model_only",
                    }
                ],
                "vectors": [None] * len(self.vector_names),
            }
            self.vector_props_written = False
            self.vector_names_for_props_written = set()
            self.num_bytes_per_example = 0
        self.output_properties_file = open("{}.vecp".format(path_with_basename), mode="w")
        self.num_records = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _get_vector_length(dims):
        return reduce(mul, dims, 1)

    def append(self, example_indices, tensors, inverse_logit=False):
        num_rows = None
        if isinstance(tensors, Variable):
            tensors = (tensors,)
        for tensor_id, tensor_pytorch in enumerate(tensors):
            if inverse_logit:
                # Pytorch tensors output logits, inverse of logistic function (1 / 1 + exp(-z))
                # Take inverse of logit (exp(logit(z)) / (exp(logit(z) + 1)) to get logistic fn value back
                tensor_pytorch_exp = torch.exp(tensor_pytorch)
                tensor_pytorch = torch.div(tensor_pytorch_exp, torch.add(tensor_pytorch_exp, 1))
            tensor = tensor_pytorch.data.to(self.device).numpy()
            num_rows_tensor = tensor.shape[0]
            if num_rows is not None and num_rows_tensor != num_rows:
                raise RuntimeError("Shape mismatch")
            num_rows = num_rows_tensor
            tensor_dtype = tensor.dtype
            if tensor_dtype == np.dtype("float32"):
                fmt_string_type = "f"
                vector_bytes_per_element = 4
            else:
                raise NotImplementedError
            vector_length = (self.vector_lengths[self.vector_names[tensor_id]]
                             if self.using_input_data
                             else VectorWriterBinary._get_vector_length(tensor[0].shape))
            fmt_string = ">IQII{}{}".format(vector_length, fmt_string_type)
            if not self.using_input_data and not self.vector_props_written:
                self.num_bytes_per_example += vector_length * vector_bytes_per_element + self.header_size
            for row in range(num_rows):
                flattened_row = np.ravel(tensor[row], "C")
                self.vec_file.write(struct.pack(fmt_string,
                                                self.sample_id,
                                                example_indices[row],
                                                tensor_id,
                                                len(flattened_row),
                                                *flattened_row))
            if not self.using_input_data and not self.vector_props_written:
                self.output_properties["vectors"][tensor_id] = {
                    "vectorName": self.vector_names[tensor_id],
                    "vectorDimension": tensor[0].shape,
                    "vectorType": str(tensor_dtype),
                    "vectorElementSize": vector_bytes_per_element,
                    "vectorNumBytesForElements": vector_bytes_per_element * vector_length
                }
                self.vector_names_for_props_written.add(self.vector_names[tensor_id])
                if self.vector_names_for_props_written == set(self.vector_names):
                    self.vector_props_written = True
                    self.output_properties["numBytesPerExample"] = self.num_bytes_per_example
            self.num_records += num_rows

    def close(self):
        num_bytes_written = self.vec_file.tell()
        expected_num_bytes = self.num_records * self.num_bytes_per_example
        if num_bytes_written != expected_num_bytes:
            warnings.warn("Warning: num bytes written {} differs from expected {}".format(num_bytes_written,
                                                                                          expected_num_bytes))
        self.vec_file.close()
        self.output_properties["numRecords"] = self.num_records
        json.dump(self.output_properties, self.output_properties_file, indent=4)
        self.output_properties_file.close()
