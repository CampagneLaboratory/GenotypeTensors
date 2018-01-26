import struct

from org.campagnelab.dl.genotypetensors.VectorReaderBase import VectorReaderBase, VectorLine

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
        # Get total number of bytes in file by going to end of file, checking position, and returning to start
        self.vector_fp.seek(0, 2)
        self.num_bytes = self.vector_fp.tell()
        self.vector_fp.seek(0, 0)
        expected_bytes_in_file = self.vector_properties.num_records * self.vector_properties.num_bytes_per_example
        if self.num_bytes != expected_bytes_in_file:
            error_msg = ("Bytes in file {} not equal to expected number of bytes {} "
                         "based on {} examples and {} bytes per example")
            raise ValueError(error_msg.format(self.num_bytes, expected_bytes_in_file,
                                              self.vector_properties.num_records,
                                              self.vector_properties.num_bytes_per_example))

    def get_next_vector_line(self):
        if self.vector_fp.tell() == self.num_bytes:
            raise StopIteration
        line_sample_id = self._get_next_value("int")
        line_example_id = self._get_next_value("long")
        line_vector_id = self._get_next_value("int")
        num_line_vector_elements = self._get_next_value("int", numpy_convert=False)
        line_element_type = self.vector_properties.get_vector_type_from_idx(line_vector_id)
        line_vector_elements = self._get_next_value(line_element_type, num_elements=num_line_vector_elements)
        return VectorLine(line_example_id, line_sample_id, line_vector_id, line_vector_elements)

    def _get_next_value(self, data_type, num_elements=1, numpy_convert=True):
        if data_type == "int":
            dtype = np.uint32 if numpy_convert else int
            fmt_string = ">{}I".format(num_elements)
            unpacked_value = struct.unpack(fmt_string, bytes(self.vector_fp.read(4 * num_elements)))
        elif data_type == "long":
            dtype = np.uint64 if numpy_convert else int
            fmt_string = ">{}Q".format(num_elements)
            unpacked_value = struct.unpack(fmt_string, bytes(self.vector_fp.read(8 * num_elements)))
        elif data_type == "float32":
            dtype = np.float32 if numpy_convert else float
            fmt_string = ">{}f".format(num_elements)
            unpacked_value = struct.unpack(fmt_string, bytes(self.vector_fp.read(4 * num_elements)))
        else:
            raise ValueError("Unknown data type to unpack: {}".format(data_type))
        if type(unpacked_value) != tuple or len(unpacked_value) != num_elements:
            raise ValueError("Error in reading in binary data")
        return dtype(unpacked_value[0]) if num_elements == 1 else dtype(unpacked_value)

    def set_to_example_at_idx(self, idx):
        if idx < 0:
            raise ValueError("Index must be positive")
        elif idx >= self.vector_properties.num_records:
            raise ValueError("Index greater than the maximum possible index, {}"
                             .format(self.vector_properties.num_records - 1))
        else:
            self.vector_fp.seek(idx * self.vector_properties.num_bytes_per_example, 0)

    def close(self):
        self.vector_fp.close()
