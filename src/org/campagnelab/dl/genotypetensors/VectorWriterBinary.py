import copy
import json
import os
import struct

from functools import reduce
from operator import mul

from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader

import numpy as np


class VectorWriterBinary:
    major_version = 0
    minor_version = 4

    example_id_size = 8
    vector_id_size = 4
    sample_id_size = 4
    vector_length_size = 4
    header_size = example_id_size + vector_id_size + sample_id_size + vector_length_size
    vector_element_size = 4

    def __init__(self, path_with_basename, sample_id, tensor_names, input_data_path=None):
        self.basename = path_with_basename
        self.vec_file = open(self.basename + ".vec", "wb")
        self.sample_id = sample_id
        self.vector_names = tensor_names
        self.output_properties = {}
        self.using_input_data = input_data_path is not None
        if self.using_input_data:
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
        else:
            self.output_properties = {
                "majorVersion": VectorWriterBinary.major_version,
                "minorVersion": VectorWriterBinary.minor_version,
                "fileType": "binary",
                "domainDescriptor": "pytorch_model_only",
                "featureMapper": "pytorch_model_only",
                "headerSize": VectorWriterBinary.header_size,
                "inputFiles": ["pytorch_model_only"],
                "samples": [
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

    def append(self, example_index, tensors):
        num_rows = None
        for tensor_id, tensor_pytorch in enumerate(tensors):
            tensor = tensor_pytorch.data.cpu().numpy()
            num_rows_tensor = tensor.shape[0]
            if num_rows is not None and num_rows_tensor != num_rows:
                raise RuntimeError("Shape mismatch")
            num_rows = num_rows_tensor
            tensor_dtype = tensor.dtype
            if tensor_dtype == np.dtype("float32"):
                fmt_string_type = "F"
                vector_bytes_per_element = 4
            else:
                raise NotImplementedError
            if self.using_input_data:
                fmt_string = ">IQII{}{}".format(self.vector_lengths[self.vector_names[tensor_id]], fmt_string_type)
            else:
                vector_length = VectorWriterBinary._get_vector_length(tensor.shape)
                fmt_string = ">IQII{}{}".format(vector_length, fmt_string_type)
                if not self.vector_props_written:
                    self.num_bytes_per_example += vector_length * vector_bytes_per_element
            for row in range(num_rows):
                flattened_row = np.ravel(tensor[row], "C")
                self.vec_file.write(struct.pack(fmt_string,
                                                self.sample_id,
                                                example_index,
                                                tensor_id,
                                                len(flattened_row),
                                                *flattened_row))
            if not self.using_input_data and not self.vector_props_written:
                self.output_properties["vectors"][tensor_id] = {
                    "vectorName": self.vector_names[tensor_id],
                    "vectorDimension": tensor.shape,
                    "vectorType": str(tensor_dtype),
                    "vectorNumBytesPerElement": vector_bytes_per_element
                }
                self.vector_names_for_props_written.add(self.vector_names[tensor_id])
                if self.vector_names_for_props_written == set(self.vector_names):
                    self.vector_props_written = True
                    self.output_properties["numBytesPerExample"] = self.num_bytes_per_example
            self.num_records += 1

    def close(self):
        self.vec_file.close()
        self.output_properties["numRecords"] = self.num_records
        json.dump(self.output_properties, self.output_properties_file, indent=4)
        self.output_properties_file.close()
