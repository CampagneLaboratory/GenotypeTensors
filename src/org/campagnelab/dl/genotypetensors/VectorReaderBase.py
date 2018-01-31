from abc import ABC, abstractmethod


class VectorReaderBase(ABC):
    def __init__(self, path_to_vector, vector_properties):
        self.path_to_vector = path_to_vector
        if not path_to_vector.endswith(".vec"):
            self.path_to_vector += ".vec"
        self.vector_properties = vector_properties

    def get_vector_properties(self):
        return self.vector_properties

    @abstractmethod
    def get_next_vector_line(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def set_to_example_at_idx(self, idx):
        pass


class VectorLine:
    def __init__(self, line_example_id, line_sample_id, line_vector_id, line_vector_elements):
        self.line_example_id = line_example_id
        self.line_sample_id = line_sample_id
        self.line_vector_id = line_vector_id
        self.line_vector_elements = line_vector_elements
