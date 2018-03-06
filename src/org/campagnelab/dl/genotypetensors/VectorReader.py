import argparse

import sys

import itertools
import threading

from org.campagnelab.dl.genotypetensors.VectorPropertiesReader import VectorPropertiesReader
from org.campagnelab.dl.genotypetensors.VectorReaderBinary import VectorReaderBinary
from org.campagnelab.dl.genotypetensors.VectorReaderText import VectorReaderText

import os

from multiprocessing import current_process


class VectorReader:
    def __init__(self, path_to_vector, sample_id, vector_names, assert_example_ids=False, return_example_id=False):
        """
        :param path_to_vector: Path to the .vec file.
        :param sample_id: sample_id to read vectors from
        :param vector_names: names of vector VectorReader should read
        :param assert_example_ids: If True, test that example ids never repeat.
        :param return_example_id: If True, return the example id as the first element of the tuple
        """
        basename, file_extension = os.path.splitext(path_to_vector)
        properties_path = "{}.vecp".format(basename)
        self.path_to_vector = "{}.vec".format(basename)
        self.vector_reader_properties = VectorPropertiesReader(properties_path)
        self.sample_id = sample_id
        self.vector_names = vector_names
        self.vector_ids = [self.vector_reader_properties.get_vector_idx_from_name(vector_name)
                           for vector_name in self.vector_names]
        self.assert_example_ids = assert_example_ids
        self.return_example_id = return_example_id
        # All of the possible vector and sample ID combinations for a record
        self.sample_vector_ids = set(itertools.product(range(len(self.vector_reader_properties.samples)),
                                                       range(len(self.vector_reader_properties.vectors))))
        if self.assert_example_ids:
            self.processed_example_ids = set([])
        version_number = self.vector_reader_properties.get_version_number()
        if version_number[0] == 0 and version_number[1] < 2:
            raise ValueError("Version number too low to be parsed by reader")
        vector_file_type = self.vector_reader_properties.file_type
        if vector_file_type == "text" or vector_file_type == "gzipped+text":
            self.vector_reader = VectorReaderText(self.path_to_vector, self.vector_reader_properties)
        elif vector_file_type == "binary":
            self.vector_reader = VectorReaderBinary(self.path_to_vector, self.vector_reader_properties)
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        curr_example = None
        processed_sample_vector_ids = set()
        example_ids = []
        for _ in range(len(self.sample_vector_ids)):
            next_vector_line = self.vector_reader.get_next_vector_line()
            # print("Process {} vector line info {}".format(current_process().pid, next_vector_line))
            example_ids.append(next_vector_line.line_example_id)
            if curr_example is None:
                curr_example = ExampleVectorLines(next_vector_line.line_example_id, self.vector_ids, self.sample_id)
                if self.assert_example_ids and curr_example.example_id in self.processed_example_ids:
                    raise RuntimeError("Example ID % already processed".format(
                                next_vector_line.line_example_id))
            if curr_example.same_example(next_vector_line.line_example_id):
                curr_example.add_vector_line(next_vector_line)
                processed_sample_vector_ids.add((next_vector_line.line_sample_id, next_vector_line.line_vector_id))
            else:
                break
        if not processed_sample_vector_ids == self.sample_vector_ids:
            unprocessed_sample_vector_ids = self.sample_vector_ids - processed_sample_vector_ids
            # print("ERR process {} example ids {}".format(current_process().pid, example_ids))
            raise Exception(
                "process= {} vector-names= {} vector-ids= {} sample-index={}\n"
                "All sample-vector-ids={} processed sample-vector-ids={}\n"
                "Missing sample index-vector index pairs for example {}: {}".format(
                    current_process().pid,
                    str(self.vector_names),
                    str(self.vector_ids),
                    str(self.sample_id),
                    str(self.sample_vector_ids),
                    str(processed_sample_vector_ids),
                    curr_example.example_id,
                    unprocessed_sample_vector_ids
                )
            )
        else:
            return curr_example.get_tuples(self.return_example_id)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.vector_reader.close()

    def set_to_example_at_idx(self, idx):
        # assert 0 <= idx < self.length, "index out of bounds."
        if self.vector_reader_properties.file_type != "binary":
            raise ValueError("Operation only valid for binary files")
        else:
            self.vector_reader.set_to_example_at_idx(idx)


class ExampleVectorLines:
    def __init__(self, example_id, vector_ids, sample_id):
        self.example_id = example_id
        self.vector_lines = {}
        self.vector_ids = vector_ids
        self.sample_id = sample_id

    def set_example_id(self, example_id):
        self.example_id = example_id

    def same_example(self, other_example_id):
        return self.example_id == other_example_id

    def add_vector_line(self, vector_line):
        if vector_line.line_sample_id == self.sample_id:
            self.vector_lines[vector_line.line_vector_id] = vector_line.line_vector_elements

    def get_tuples(self, return_example_id):
        try:
            vector_lines = [self.vector_lines[vector_id] for vector_id in self.vector_ids]
        except KeyError:
            vectors_missing = list(frozenset(self.vector_ids) - frozenset(self.vector_lines.keys()))
            raise RuntimeError("Vectors missing for example id {}: {}".format(self.example_id, vectors_missing))
        if return_example_id:
            return tuple([self.example_id] + vector_lines)
        else:
            return tuple(vector_lines)


if __name__ == "__main__":
    def _get_example_vector_line_str(example_to_print, vector_names):
        example_str = "Example ID: {}\n".format(example_to_print[0])
        all_vector_values = enumerate(example_to_print[1:])
        for vector_idx, vector_values in all_vector_values:
            example_str += "{}: {}\n".format(vector_names[vector_idx], vector_values)
        return example_str + "\n"


    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", help=".vec or .vec.gz file to read in", type=str, required=True)
    arg_parser.add_argument("-s", "--sample-id", help="sample ID to use", type=int, default=0)
    arg_parser.add_argument("-v", "--vector_names", help="vector names to use", type=str, nargs="+")
    arg_parser.add_argument("-l", "--limit", help="Maximum number of examples to print", type=int, default=sys.maxsize)
    arg_parser.add_argument("-o", "--output", help="Output file to write out example info to. If none, print to stdout",
                            type=str)
    args = arg_parser.parse_args()
    output_file = None
    if args.output is not None:
        output_file = open(args.output, "w")
    with VectorReader(args.input, sample_id=args.sample_id, vector_names=args.vector_names,
                      assert_example_ids=True, return_example_id=True) as vector_reader:
        i = 0
        for next_example in vector_reader:
            if i < args.limit:
                if output_file is not None:
                    output_file.write(_get_example_vector_line_str(next_example, args.vector_names))
                else:
                    print(_get_example_vector_line_str(next_example, args.vector_names))
                i += 1
            else:
                break
