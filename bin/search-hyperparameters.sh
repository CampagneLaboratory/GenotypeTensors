#!/usr/bin/env bash
. `dirname "${BASH_SOURCE[0]}"`/setup.sh
set -x
echo "usage: search-hyperparameters [num-models] --mode semisupervised|supervised-direct|supervised-mixup --problem genotyping:basename --num-training N ..."
num_executions=$1
shift
if [ -z "${SBI_SEARCH_PARAM_CONFIG+set}" ]; then
    echo "Variable SBI_SEARCH_PARAM_CONFIG not defined, you must specify a command file manually."
    OPTION_COMMAND=" "
else
    COMMANDS="gen-args-${RANDOM}.txt"
    arg-generator.sh 1g --config ${SBI_SEARCH_PARAM_CONFIG} --output ${COMMANDS} --num-commands ${num_executions}
    OPTION_COMMAND="--commands ${COMMANDS}"
fi


python "${GENOTYPE_TENSORS}/src/org/campagnelab/dl/genotypetensors/autoencoder/SearchHyperparameters.py" "$@" ${OPTION_COMMAND}
