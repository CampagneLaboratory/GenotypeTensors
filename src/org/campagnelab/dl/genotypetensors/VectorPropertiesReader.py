import argparse
import json


class VectorPropertiesReader:
    def __init__(self, path_to_vector):
        with open(path_to_vector, 'r') as reader_fp:
            self.vector_properties = json.load(reader_fp)
            self.major_version = self.vector_properties["majorVersion"]
            self.minor_version = self.vector_properties["minorVersion"]
            self.file_type = self.vector_properties["fileType"]
            self.samples = self.vector_properties["samples"]
            self.vectors = self.vector_properties["vectors"]
            self.num_records = self.vector_properties["numRecords"]

    def get_version_number(self):
        return self.major_version, self.minor_version

    def get_vector_file_type(self):
        return self.file_type

    def get_samples(self):
        return self.samples

    def get_sample_idx(self, idx):
        return self.samples[idx]

    def get_sample_name_idx(self, idx):
        return self.samples[idx]["sampleName"]

    def get_sample_type_idx(self, idx):
        return self.samples[idx]["sampleType"]

    def get_vectors(self):
        return self.vectors

    def get_vector_name_idx(self, idx):
        return self.vectors[idx]["vectorName"]

    def get_vector_type_idx(self, idx):
        return self.vectors[idx]["vectorType"]

    def get_vector_dimensions_idx(self, idx):
        return self.vectors[idx]["vectorDimension"]

    def get_num_records(self):
        return self.num_records


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", help=".vecp file to read in", type=str, required=True)
    args = arg_parser.parse_args()
    vec_properties_reader = VectorPropertiesReader(args.input)
    version_number = ".".join(map(str, vec_properties_reader.get_version_number()))
    print("Version number: {}".format(version_number))
    print("File type: {}".format(vec_properties_reader.get_vector_file_type()))
    print("Sample 0 name: {}".format(vec_properties_reader.get_sample_name_idx(0)))
    print("Sample 0 type: {}".format(vec_properties_reader.get_sample_type_idx(0)))
    print("Vector 0 name: {}".format(vec_properties_reader.get_vector_name_idx(0)))
    print("Vector 0 type: {}".format(vec_properties_reader.get_vector_type_idx(0)))
    print("Vector 0 dim: {}".format(vec_properties_reader.get_vector_dimensions_idx(0)))
