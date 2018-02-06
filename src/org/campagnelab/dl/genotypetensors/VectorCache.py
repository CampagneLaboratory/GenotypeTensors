import argparse
import copy
import json
import os
import struct
import sys

from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader
from org.campagnelab.dl.genotypetensors.VectorReaderText import VectorReaderText
from org.campagnelab.dl.utils.utils import progress_bar


class VectorCache:
    def __init__(self, path_to_vector, max_records=sys.maxsize):
        self.path_basename, _ = os.path.splitext(path_to_vector)
        properties_path = "{}.vecp".format(self.path_basename)
        self.vector_reader_properties = VectorPropertiesReader(properties_path)
        self.vector_text_reader = VectorReaderText(path_to_vector, self.vector_reader_properties)
        self.output_path = "{}-cached.vec".format(self.path_basename)
        self.max_records = min(self.vector_reader_properties.num_records, max_records)
        self.num_vector_lines_per_example = (len(self.vector_reader_properties.samples)
                                             * len(self.vector_reader_properties.vectors))
        self.total_vector_lines = self.num_vector_lines_per_example * self.max_records
        self.output_writer = open(self.output_path, "wb")
        self.cache_output_properties = copy.deepcopy(self.vector_reader_properties.vector_properties)
        self.cache_output_properties["fileType"] = "binary"
        self.expected_bytes = (self.max_records
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
        num_vector_lines = 0
        try:
            while num_vector_lines < self.total_vector_lines:
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
                num_vector_lines += 1
                num_examples_written = num_vector_lines / self.num_vector_lines_per_example
                if (num_vector_lines % self.num_vector_lines_per_example == 0) and (num_examples_written % 1000 == 1):
                    progress_bar(num_examples_written,
                                 self.max_records, "Caching " + self.path_basename)

        except StopIteration:
            pass
        self.output_writer.flush()
        self.output_writer.seek(0, 2)
        num_bytes_written = self.output_writer.tell()
        if num_bytes_written != self.expected_bytes:
            raise RuntimeError(
                "Number of bytes written {} differs from expected {}. "
                "Wrote {} vector lines, {} per example "
                "for {} records, {} stray lines.".format(num_bytes_written,
                                                         self.expected_bytes,
                                                         num_vector_lines,
                                                         self.num_vector_lines_per_example,
                                                         num_vector_lines / self.num_vector_lines_per_example,
                                                         num_vector_lines % self.num_vector_lines_per_example))
        self.close()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", help="input .vec file to read in", type=str, required=True)
    args = arg_parser.parse_args()
    with VectorCache(args.input) as vector_cache:
        vector_cache.write_lines()
