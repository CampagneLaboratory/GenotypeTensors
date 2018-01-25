#!/usr/bin/env bash
. `dirname "${BASH_SOURCE[0]}"`/setup.sh
set -x
echo ${GENOTYPE_TENSORS}

python "${GENOTYPE_TENSORS}/src/org/campagnelab/dl/genotypetensors/autoencoder/TrainAutoEncoder.py" "$@"
