
class VectorWriter:
    def __init__(self, path_with_basename):
        self.basename=path_with_basename

    def __enter__(self):
        self.vec_file=open(self.basename+".vec")
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.vec_file.close()

    def append(self,example_index, tensors):
        self.vec_file.write(str(example_index))
        self.vec_file.write(" ")
        tensor_index=0
        for tensor in tensors:
            self.vec_file.write(tensor_index)
            self.vec_file.write(" ")
            self.vec_file.write(tensor.join(" "))
            tensor_index+=1

        self.vec_file.write("\n")

