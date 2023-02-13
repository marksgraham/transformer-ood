#!/bin/bash
#
# A simple script to build the distributed Docker image.
#
# $ create_docker_image.sh
set -ex
TAG=transformer-ood

# get monai generative
git clone git@github.com:Project-MONAI/GenerativeModels.git

cp ../requirements.txt .
docker build --network=host --tag "aicregistry:5000/${USER}:${TAG}" . \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}

#push
docker push "aicregistry:5000/${USER}:${TAG}"
