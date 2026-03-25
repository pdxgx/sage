#!/bin/bash

  docker run --gpus all \
  -v /your/path/data:/data \
  -v /your/path/checkpoints:/checkpoints \
  -v /your/path/logs:/logs \
  --shm-size=16g \
  --rm \
  wmschreyer/sage:latest \
  python /workspace/scripts/train_ham_resnets.py --imagedir /data/ham/images \
  --metafile /data/ham/metadata.csv \
  --savedir /checkpoints \
  --logdir /logs \

docker run --gpus all \
  -v /your/path/data:/data \
  -v /your/path/checkpoints:/checkpoints \
  -v /your/path/outputs:/outputs \
  -v /your/path/logs:/logs \
  --shm-size=16g \
  --rm \
  wmschreyer/sage:latest \
  python /workspace/scripts/resnet_eval_all.py \
  --modeldir /checkpoints \
  --datadir /data \
  --outdir /outputs \
  --logdir /logs