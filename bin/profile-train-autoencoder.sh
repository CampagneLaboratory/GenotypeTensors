#!/usr/bin/env bash
. `dirname "${BASH_SOURCE[0]}"`/setup.sh
#set -x
#echo ${GENOTYPE_TENSORS}
#echo ${PYTHONPATH}

python -m cProfile -o train_autoencoder.prof "${GENOTYPE_TENSORS}/src/org/campagnelab/dl/genotypetensors/autoencoder/TrainAutoEncoder.py" "$@"
