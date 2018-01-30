from org.campagnelab.dl.genotypetensors.VectorReaderBase import VectorReaderBase, VectorLine

import numpy as np
import gzip


class VectorReaderText(VectorReaderBase):
    def __init__(self, path_to_vector, vector_reader_properties):
        super().__init__(path_to_vector, vector_reader_properties)
        if self.vector_properties.file_type == "gzipped+text":
            self.vector_fp = gzip.open(self.path_to_vector, "rb")
        else:
            self.vector_fp = open(path_to_vector, "r")

    def get_next_vector_line(self):
        line = next(self.vector_fp)
        line_split = line.strip().split()
        line_example_id = np.uint64(line_split[1])
        line_sample_id = np.uint32(line_split[0])
        line_vector_id = np.uint32(line_split[2])
        line_vector_elements = np.array(line_split[3:], dtype=np.float32)
        line_vector_elements = np.reshape(line_vector_elements,
                                          self.vector_properties.get_vector_type_from_idx(line_vector_id),
                                          "C")
        return VectorLine(line_example_id, line_sample_id, line_vector_id, line_vector_elements)

    def close(self):
        self.vector_fp.close()

    def set_to_example_at_idx(self, idx):
        raise TypeError("Operation invalid for text files")
