import struct

from org.campagnelab.dl.genotypetensors.VectorReaderBase import VectorReaderBase, VectorLine

import numpy as np
import gzip


class VectorReaderBinary(VectorReaderBase):
    def __init__(self, path_to_vector, vector_reader_properties):
        super().__init__(path_to_vector, vector_reader_properties)
        self.vector_fp = open(path_to_vector, "rb")

    def get_next_vector_line(self):
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
            return dtype(struct.unpack(">I", bytearray(self.vector_fp.read(4))))
        elif data_type == "long":
            dtype = np.uint64 if numpy_convert else int
            return dtype(struct.unpack(">L", bytearray(self.vector_fp.read(8))))
        elif data_type == "float":
            dtype = np.float32 if numpy_convert else float
            return dtype(struct.unpack(">f", bytearray(self.vector_fp.read(4))))
        else:
            raise ValueError("Unknown data type to unpack: {}".format(data_type))

    def close(self):
        self.vector_fp.close()
