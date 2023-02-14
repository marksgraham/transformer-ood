#!/bin/bash
num_gpus=2
runai submit \
  --name vqvae \
  --image aicregistry:5000/mark:ddpm-ood \
  --backoff-limit 0 \
  --gpu ${num_gpus} \
  --cpu 12 \
  --large-shm \
  --host-ipc \
  --project mark \
  --run-as-user \
  --volume /nfs/home/mark/projects/transformer-ood/transformer-ood/:/project/ \
  --volume /nfs/home/mark/projects/generative_ct/data/nonrigid_cromis_all/images/:/data/ \
  --volume /nfs/home/mark/projects/transformer-ood/output/:/output/ \
  --command -- torchrun --nproc_per_node=${num_gpus} --nnodes=1 --node_rank=0 /project/train.py \
  --output_dir=/output/ \
  --model_name=vqvae \
  --training_dir=/data/train \
  --validation_dir=/data/val \
  --n_epochs=1000 \
  --batch_size=3 \
  --eval_freq=10 \
  --cache_data=1