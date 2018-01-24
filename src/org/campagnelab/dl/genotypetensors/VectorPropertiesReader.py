import argparse
import json


class VectorPropertiesReader:
    def __init__(self, path_to_vector_properties):
        """
        :param path_to_vector_properties: path to .vecp file
        """
        with open(path_to_vector_properties, 'r') as reader_fp:
            self.vector_properties = json.load(reader_fp)
            self.major_version = self.vector_properties["majorVersion"]
            self.minor_version = self.vector_properties["minorVersion"]
            self.file_type = self.vector_properties["fileType"]
            self.samples = self.vector_properties["samples"]
            self.vectors = self.vector_properties["vectors"]
            for vector in self.vectors:
                vector["vectorDimension"] = tuple(vector["vectorDimension"])
            self.num_records = self.vector_properties["numRecords"]
            self.vector_names_to_idx = {vector[1]["vectorName"]: vector[0] for vector in enumerate(self.vectors)}
            self.sample_names_to_idx = {sample[1]["sampleName"]: sample[0] for sample in enumerate(self.samples)}

    def get_version_number(self):
        """
        Get version number for vec file as tuple of major and minor version
        :return: (major_version, minor_version) tuple of ints
        """
        return self.major_version, self.minor_version

    def get_vector_file_type(self):
        """
        Get vector file type- either text, gzipped+text, or binary
        :return: vector file type
        """
        return self.file_type

    def get_vectors(self):
        """
        Get list of vectors, as they appear in .vecp file
        :return: list of vectors where each element is a Python dict containing vector info
        """
        return self.vectors

    def get_vector_from_idx(self, idx):
        """
        Get a vector dict at a particular vector index
        :param idx: index of vector in list of vectors from .vecp file
        :return: Python dict containing vector info
        """
        return self.vectors[idx]

    def get_vector_from_name(self, vector_name):
        """
        Get a vector dict given a particular vector name
        :param: vector_name: name of vector from .vecp file
        :return: Python dict containing vector info
        """
        return self.vectors[self.vector_names_to_idx[vector_name]]

    def get_vector_name_from_idx(self, idx):
        """
        Get the vector name corresponding to a given vector index
        :param idx: index of vector in list of vectors from .vecp file
        :return: name of vector as string
        """
        return self.vectors[idx]["vectorName"]

    def get_vector_type_from_idx(self, idx):
        """
        Get the vector type given a vector index
        :param idx: index of vector in list of vectors from .vecp file
        :return: type of vector as a string
        """
        return self.vectors[idx]["vectorType"]

    def get_vector_dimensions_from_idx(self, idx):
        """
        Get the vector dimension tuple given a vector index
        :param idx: index of vector in list of vectors from .vecp file
        :return: tuple of vector dimensions
        """
        return tuple(self.vectors[idx]["vectorDimension"])

    def get_vector_dimensions_from_name(self, vector_name):
        """
        Get the vector dimension tuple given a vector name
        :param vector_name: name of vector from .vecp file
        :return: tuple of vector dimensions
        """
        return tuple(self.vectors[self.vector_names_to_idx[vector_name]]["vectorDimension"])

    def get_vector_type_from_name(self, vector_name):
        """
        Get the vector type given a vector name
        :param vector_name: name of vector from .vecp file
        :return: type of vector as a string
        """
        return self.vectors[self.vector_names_to_idx[vector_name]]["vectorType"]

    def get_vector_idx_from_name(self, vector_name):
        """
        Get the vector index corresponding to a given vector name
        :param vector_name: name of vector from .vecp file
        :return: index of vector as int
        """
        return self.vector_names_to_idx[vector_name]

    def get_samples(self):
        """
        Get list of samples, as they appear in .vecp file
        :return: list of samples where each element is a Python dict containing sample info
        """
        return self.samples

    def get_sample_from_idx(self, idx):
        """
        Get a sample dict at a particular sample index
        :param idx: index of sample in list of samples from .vecp file
        :return: Python dict containing sample info
        """
        return self.samples[idx]

    def get_sample_from_name(self, sample_name):
        """
        Get a sample dict for a particular sample given a sample name
        :param sample_name: name of sample from .vecp file
        :return: Python dict containing sample info
        """
        return self.samples[self.sample_names_to_idx[sample_name]]

    def get_sample_name_from_idx(self, idx):
        """
        Get sample name given a sample index
        :param idx: index of sample in list of samples from .vecp file
        :return: sample name
        """
        return self.samples[idx]["sampleName"]

    def get_sample_type_from_idx(self, idx):
        """
        Get sample type given a sample index
        :param idx: index of sample in list of samples from .vecp file
        :return: sample type
        """
        return self.samples[idx]["sampleType"]

    def get_sample_type_from_name(self, sample_name):
        """
        Get the sample type given a sample name
        :param sample_name: name of sample from .vecp file
        :return: sample type
        """
        return self.samples[self.sample_names_to_idx[sample_name]]["sampleType"]

    def get_sample_idx_from_name(self, sample_name):
        """
        Get the sample index corresponding to a given sample name
        :param sample_name: name of sample from .vecp file
        :return: index of sample as int
        """
        return self.sample_names_to_idx[sample_name]

    def get_num_records(self):
        """
        Get number of records stored in .vec file
        :return: number of records
        """
        return self.num_records


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", help=".vecp file to read in", type=str, required=True)
    args = arg_parser.parse_args()
    vec_properties_reader = VectorPropertiesReader(args.input)
    version_number = ".".join(map(str, vec_properties_reader.get_version_number()))
    print("Version number: {}".format(version_number))
    print("File type: {}".format(vec_properties_reader.get_vector_file_type()))
    print("Sample 0 name: {}".format(vec_properties_reader.get_sample_name_from_idx(0)))
    print("Sample 0 type: {}".format(vec_properties_reader.get_sample_type_from_idx(0)))
    print("Vector 0 name: {}".format(vec_properties_reader.get_vector_name_from_idx(0)))
    print("Vector 0 type: {}".format(vec_properties_reader.get_vector_type_from_idx(0)))
    print("Vector 0 dim: {}".format(vec_properties_reader.get_vector_dimensions_from_idx(0)))
    sample_0_name = vec_properties_reader.get_sample_name_from_idx(0)
    print("Sample 0 idx from name: {}".format(vec_properties_reader.get_sample_idx_from_name(sample_0_name)))
    print("Sample 0 type from name: {}".format(vec_properties_reader.get_sample_type_from_name(sample_0_name)))
    vector_0_name = vec_properties_reader.get_vector_name_from_idx(0)
    print("Vector 0 idx from name: {}".format(vec_properties_reader.get_vector_idx_from_name(vector_0_name)))
    print("Vector 0 type from name: {}".format(vec_properties_reader.get_vector_type_from_name(vector_0_name)))
    print("Vector 0 dim from name: {}".format(vec_properties_reader.get_vector_dimensions_from_name(vector_0_name)))
    print("All samples: {}".format(vec_properties_reader.get_samples()))
    print("All vectors: {}".format(vec_properties_reader.get_vectors()))
