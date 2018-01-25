import struct

from org.campagnelab.dl.genotypetensors.VectorReaderBase import VectorReaderBase, VectorLine

from functools import reduce
from operator import mul

import numpy as np


class VectorReaderBinary(VectorReaderBase):
    example_id_size = 8
    vector_id_size = 4
    sample_id_size = 4
    vector_length_size = 4
    header_size = example_id_size + vector_id_size + sample_id_size + vector_length_size
    vector_element_size = 4

    def __init__(self, path_to_vector, vector_reader_properties):
        super().__init__(path_to_vector, vector_reader_properties)
        self.vector_fp = open(path_to_vector, "rb")
        # Get length of each vector type by multiplying dimensions together
        vector_lengths = (reduce(mul, self.vector_properties.get_vector_dimensions_from_idx(vector_index), 1)
                          for vector_index in range(len(self.vector_properties.get_vectors())))
        # Get total number of bytes in file by going to end of file, checking position, and returning to start
        self.vector_fp.seek(0, 2)
        self.num_bytes = self.vector_fp.tell()
        self.vector_fp.seek(0, 0)
        self.total_bytes_per_example = sum((VectorReaderBinary.header_size +
                                            (vector_length * VectorReaderBinary.vector_element_size)
                                            for vector_length in vector_lengths))
        if self.num_bytes % self.total_bytes_per_example != 0:
            raise ValueError("Bytes per example {} incompatible with total number of bytes in file {}".format(
                self.total_bytes_per_example,
                self.num_bytes
            ))
        self.num_examples = self.num_bytes / self.total_bytes_per_example

    def get_next_vector_line(self):
        if self.vector_fp.tell() == self.num_bytes:
            raise StopIteration
        line_sample_id = self._get_next_value("int")
        line_example_id = self._get_next_value("long")
        line_vector_id = self._get_next_value("int")
        num_line_vector_elements = self._get_next_value("int", numpy_convert=False)
        line_vector_elements = []
        for _ in range(num_line_vector_elements):
            line_vector_elements.append(self._get_next_value("float"))
        line_vector_elements = np.array(line_vector_elements, dtype=np.float32)
        return VectorLine(line_example_id, line_sample_id, line_vector_id, line_vector_elements)

    def _get_next_value(self, data_type, numpy_convert=True):
        if data_type == "int":
            dtype = np.uint32 if numpy_convert else int
            unpacked_value = struct.unpack(">I", bytearray(self.vector_fp.read(4)))
        elif data_type == "long":
            dtype = np.uint64 if numpy_convert else int
            unpacked_value = struct.unpack(">Q", bytearray(self.vector_fp.read(8)))
        elif data_type == "float":
            dtype = np.float32 if numpy_convert else float
            unpacked_value = struct.unpack(">f", bytearray(self.vector_fp.read(4)))
        else:
            raise ValueError("Unknown data type to unpack: {}".format(data_type))
        if type(unpacked_value) != tuple or len(unpacked_value) > 1:
            raise ValueError("Error in reading in binary data")
        return dtype(unpacked_value[0])

    def set_to_example_at_idx(self, idx):
        if idx < 0:
            raise ValueError("Index must be positive")
        elif idx >= self.num_examples:
            raise ValueError("Index greater than the maximum possible index, {}".format(self.num_examples - 1))
        else:
            self.vector_fp.seek(idx * self.total_bytes_per_example, 0)

    def close(self):
        self.vector_fp.close()
