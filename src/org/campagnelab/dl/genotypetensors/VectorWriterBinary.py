import copy
import json
import os
import struct

from functools import reduce
from operator import mul

from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader

import numpy as np


class VectorWriterBinary:
    # TODO: REMOVE input_data_path  we don't generally have properties to copy to the destination.
    def __init__(self, path_with_basename, sample_id, tensor_names, input_data_path):
        def _get_vector_length(vector_name_for_dims):
            return reduce(mul,
                          self.input_vector_properties_reader.get_vector_dimensions_from_name(vector_name_for_dims), 1)
        self.basename = path_with_basename
        self.vec_file = open(self.basename + ".vec", "wb")
        self.sample_id = sample_id
        self.input_vector_properties_reader = VectorPropertiesReader(
            "{}.vecp".format(input_data_path.split(os.extsep)[0])
        )
        input_properties = self.input_vector_properties_reader.vector_properties
        self.vector_names = tensor_names
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
            ]
        }
        num_bytes_per_example = (vector["vectorNumBytesPerElements"] for vector in self.output_properties["vectors"])
        num_bytes_per_example = (num_bytes_per_vector + self.output_properties["headerSize"]
                                 for num_bytes_per_vector in num_bytes_per_example)
        num_bytes_per_example = sum((num_bytes_per_vector * len(self.output_properties["samples"])
                                     for num_bytes_per_vector in num_bytes_per_example))
        self.output_properties["numBytesPerExample"] = num_bytes_per_example
        self.output_properties_file = open("{}.vecp".format(path_with_basename))
        self.vector_lengths = {vector_name: _get_vector_length(vector_name) for vector_name in tensor_names}
        self.num_records = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
            else:
                raise NotImplementedError
            fmt_string = ">IQII{}{}".format(self.vector_lengths[self.vector_names[tensor_id]], fmt_string_type)
            for row in range(num_rows):
                flattened_row = np.ravel(tensor[row], "C")
                self.vec_file.write(struct.pack(fmt_string,
                                                self.sample_id,
                                                example_index,
                                                tensor_id,
                                                len(flattened_row),
                                                *flattened_row))
            self.num_records += 1

    def close(self):
        self.vec_file.close()
        self.output_properties["numRecords"] = self.num_records
        json.dump(self.output_properties, self.output_properties_file, indent=4)
        self.output_properties_file.close()
