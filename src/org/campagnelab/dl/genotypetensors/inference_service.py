import os
import tempfile

import torch

from org.campagnelab.dl.genotypetensors.VectorReader import VectorReader
from org.campagnelab.dl.genotypetensors.VectorWriter import VectorWriter

loaded_models={}
def load_model(checkpoint_dir_path, model_name, model_label="best"):
    print('==> Loading model {} from checkpoint directory {}'.format(checkpoint_dir_path, model_name))
    model = None
    assert os.path.isdir(checkpoint_dir_path), 'Error: no checkpoint directory found!'
    checkpoint = None
    try:

        checkpoint_filename = './{}/pytorch_{}_{}.t7'.format(checkpoint_dir_path, model_name, model_label)

        checkpoint = torch.load(checkpoint_filename)
        model=checkpoint['model']

    except                FileNotFoundError:
        print("Unable to load model {} from checkpoint dir {}".format(model_name, checkpoint_dir_path))
        return None
    return model

def infer(model_name, model_label, checkpoint_dir_path, input_data_path, sample_id, tensor_names):
    """
    Perform inference using a model (identified by model_name and checkpoint_dir_path) for data
    in input_data_path (.vec/.vecp format) and returns the result in .vec/.vecp format.
    :param model_name: Name of the model to predict with.
    :param model_label: Label of the model, e.g., best or latest.
    :param checkpoint_dir_path: path of the directory where the models checkpoints are stored.
    :param input_data_path: path to the .vec file (or path with basename of .vec file).
    :param sample_id: sample name for which predictions are needed.
    :param tensor_names: list with names of tensors to feed to the model as input.
    :return: path to .vec file containing the infered/predicted examples.
    """
    global loaded_models
    model=loaded_models[model_name]
    if model is None:

        loaded_models[model_name]=model=loaded_models(checkpoint_dir_path, model_name)
    iterator=VectorReader(input_data_path,sample_id=sample_id,vector_names=tensor_names, return_example_id=True)
    output_file_path=tempfile.mkdtemp(prefix='predicted_results_')

    with VectorWriter(output_file_path) as writer:

        for example_id, data in iterator:

            output=model(data)
            writer.append(example_id, output)

    return output_file_path

def infer_genotyping(model_name, checkpoint_dir_path, input_data_path):
    infer(model_name,checkpoint_dir_path,input_data_path,sample_id=0, tensor_names=["input"])

def infer_somatic(model_name, checkpoint_dir_path, input_data_path):
        infer(model_name, checkpoint_dir_path, input_data_path, sample_id=0, tensor_names=["input"])

def ping():
    return "Alive."