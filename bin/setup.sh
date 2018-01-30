#!/usr/bin/env bash
#set -x
export GENOTYPE_TENSORS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
export PYTHONPATH="${PYTHONPATH}:${GENOTYPE_TENSORS}/src"
if [[ $OSTYPE == "cygwin" ]]; then
    GENOTYPE_TENSORS=`cygpath -m "${GENOTYPE_TENSORS}"`
    PYTHONPATH=`cygpath -m "${PYTHONPATH}"`
fi

