#!/usr/bin/env bash
. `dirname "${BASH_SOURCE[0]}"`/setup.sh

# Use this command to start an inference service.
# The inference service can be called from Java to predict with pytorch models and get the
# results back.
zerorpc --server --bind tcp://*:1234 org.campagnelab.dl.genotypetensors.inference_service
