#GenotypeTensors
This project provides tools to load the vectorized genotype information files (.vec/.vecp) produced with goby3 
and variation analysis. It also demonstrates how to train deep-learning models using information in these files 
with pytorch.

# Installation

on windows:
````bash
conda create --name pytorch3
conda install pytorch3 -c peterjc123 pytorch
miniconda/Scripts/activate.bat pytorch3
````
Use the pip.exe in miniconda for the following.

on mac:
````bash
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
````
Common to all platforms:
````bash
pip install torchvision
pip install git+https://github.com/pytorch/tnt.git@master
export UREG=<install-dir>/ureg/
````


## Example Training
Assuming you have downloaded a training dataset called dataset-2018-01-16 (with files dataset-2018-01-16-train.vec*, 
dataset-2018-01-16-validation.vec*), you can run the following to train an auto-encoder:

```bash

bin/train-autoencoder.sh --mode autoencoder \
        --problem genotyping:dataset-2018-01-16
        --lr 0.001
        --L2 1E-6
        --mini-batch-size 128 
        --checkpoint-key GENOTYPE_AUTOENCODER_1
```