# GenotypeTensors
This project provides tools to load the vectorized genotype information files (.vec/.vecp) produced with [goby3](https://github.com/CampagneLaboratory/goby3)
and [variationanalysis](https://github.com/CampagneLaboratory/). It also demonstrates how to train deep-learning models using information in these files
with pytorch.

# Installation

GenotypeTensors has been upgraded to pytorch 0.4.0.

on windows:
````bash
conda create --name pytorch3
conda install pytorch -c pytorch
miniconda/Scripts/activate.bat pytorch3
````
Use the pip.exe in miniconda for the following.

on mac:
````bash
conda install pytorch torchvision -c pytorch
````

Common to all platforms:
````bash
pip install torchvision
pip install git+https://github.com/pytorch/tnt.git@master
pip install scipy
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
project (and by the DNANexus **Convert Somatic .sbi to Tensors** app).


## Training genotyping models with semi-supervised training:

```
bin/train-autoencoder.sh --mode semisupervised_genotypes \
                --problem genotyping:/data/gen/CNG-NA12878-realigned-2018-01-30 \
                --lr 0.01 --L2 1E-6 --mini-batch-size 100 \
                --checkpoint-key GENOTYPE_SEMISUP_1 \
                --max-epochs 200 -n 500 -x 10000
```