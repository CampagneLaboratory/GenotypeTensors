# GenotypeTensors
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
        --problem genotyping:dataset-2018-01-16 \
        --lr 0.001  \
        --L2 1E-6   \
        --mini-batch-size 128 \
        --checkpoint-key GENOTYPE_AUTOENCODER_1 \
        --max-epochs 20
```

The model will be trained for 20 epochs. 
Best models are saved as checkpoints under the checkpoint directory, using the provided --checkpoint-key.

You can monitor the performance metrics during training with these files:
- all-perfs-GENOTYPE_AUTOENCODER_1.tsv
- best-perfs-GENOTYPE_AUTOENCODER_1.tsv (restricted to performance of best models, up to latest training epoch.)
- args-GENOTYPE_AUTOENCODER_1 (contains exact command line used to train the model, useful for reproducing previous runs,
    includes random seed)

If you do not provide --checkpoint-key argument, a random one is generated and saved in args-*.
This is convenient to perform hyperparameter searches.

## Training somatic models
Instead of training an auto-encoder, the code base also supports training a model to call somatic mutations. 
The vec files must have been created with a somatic feature mapper and in this case, you can do:


```bash

bin/train-autoencoder.sh --mode supervised_somatic \
        --problem somatic:dataset2-2018-01-17 \
        --lr 0.001  \
        --L2 1E-6   \
        --mini-batch-size 128 \
        --checkpoint-key GENOTYPE_AUTOENCODER_1 \
        --max-epochs 20
```

Note that we changed both the mode (now supervised_somatic) and the the dataset, 
now somatic:dataset2. Training a somatic supervised model requires specific outputs in 
the .vec files, which are produced by somatic feature mappers in the variationanalysis 
project (and by the DNANexus sbi to somatic vec app).
