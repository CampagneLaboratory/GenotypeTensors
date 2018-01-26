import argparse
import copy
import json
import os
import struct

from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader
from org.campagnelab.dl.genotypetensors.VectorReaderText import VectorReaderText


class VectorCache:
    def __init__(self, path_to_vector):
        self.path_basename = path_to_vector.split(os.extsep)[0]
        properties_path = "{}.vecp".format(self.path_basename)
        self.vector_reader_properties = VectorPropertiesReader(properties_path)
        self.vector_text_reader = VectorReaderText(path_to_vector, self.vector_reader_properties)
        self.output_path = "{}-cached.vec".format(self.path_basename)
        self.output_writer = open(self.output_path, "wb")
        self.cache_output_properties = copy.deepcopy(self.vector_reader_properties.vector_properties)
        self.cache_output_properties["fileType"] = "binary"
        self.expected_bytes = (self.vector_reader_properties.num_records
                               * self.vector_reader_properties.num_bytes_per_example)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.output_writer.close()
        self.vector_text_reader.close()
        with open("{}-cached.vecp".format(self.path_basename), "w") as vecp_fp:
            json.dump(self.cache_output_properties, vecp_fp, indent=4)

    def write_lines(self):
        try:
            while True:
                next_vector_line = self.vector_text_reader.get_next_vector_line()
                next_vector_type = self.vector_reader_properties.get_vector_type_from_idx(
                    next_vector_line.line_vector_id)
                if next_vector_type == "float32":
                    packed_type = "f"
                else:
                    raise ValueError("Unknown data type to unpack: {}".format(next_vector_type))
                fmt_string = ">IQII{}{}".format(len(next_vector_line.line_vector_elements), packed_type)
                self.output_writer.write(struct.pack(fmt_string,
                                                     next_vector_line.line_sample_id,
                                                     next_vector_line.line_example_id,
                                                     next_vector_line.line_vector_id,
                                                     len(next_vector_line.line_vector_elements),
                                                     *next_vector_line.line_vector_elements))

        except StopIteration:
            self.output_writer.seek(0, 2)
            num_bytes_written = self.output_writer.tell()
            if num_bytes_written != self.expected_bytes:
                raise RuntimeError("Number of bytes written {} differs from expected {}".format(num_bytes_written,
                                                                                                self.expected_bytes))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", help="text .vec.gz file to read in", type=str, required=True)
    args = arg_parser.parse_args()
    with VectorCache(args.input) as vector_cache:
        vector_cache.write_lines()
