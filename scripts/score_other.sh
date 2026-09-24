#!/bin/bash

docker run --gpus all \
  -v /your/path/data:/data \
  -v /your/path/checkpoints:/checkpoints \
  -v /your/path/outputs:/outputs \
  -v /your/path/logs:/logs \
  --shm-size=16g \
  --rm \
  wmschreyer/sage:latest \
  python /workspace/scripts/sage_score_other.py \
  --encoder ResNet \
  --dim 256 \
  --modeldir /checkpoints \
  --datadir /data \
  --compare your_dataset_name \
  --outdir /outputs \
  --logdir /logs \

  docker run --gpus all \
  -v /your/path/data:/data \
  -v /your/path/checkpoints:/checkpoints \
  -v /your/path/outputs:/outputs \
  -v /your/path/logs:/logs \
  --shm-size=16g \
  --rm \
  wmschreyer/sage:latest \
  python /workspace/scripts/resnet_eval_other.py \
  --modeldir /checkpoints \
  --datadir /data \
  --compare your_dataset_name \
  --outdir /outputs \
  --logdir /logs \