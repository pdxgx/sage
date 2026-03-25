#!/bin/bash

docker run --gpus all \
  -v /your/path/data:/data \
  -v /your/path/checkpoints:/checkpoints \
  -v /your/path/logs:/logs \
  --shm-size=16g \
  --rm \
  wmschreyer/sage:latest \
  python /workspace/scripts/train_ham.py --imagedir /data/ham/images \
  --metafile /data/ham/metadata.csv \
  --savedir /checkpoints \
  --logdir /logs \
  --encoder ResNet \
  --dim 256

  docker run --gpus all \
  -v /your/path/data:/data \
  -v /your/path/checkpoints:/checkpoints \
  -v /your/path/outputs:/outputs \
  -v /your/path/logs:/logs \
  --shm-size=16g \
  --rm \
  wmschreyer/sage:latest \
  python /workspace/scripts/sage_score_all.py \
  --encoder ResNet \
  --dim 256 \
  --modeldir /checkpoints \
  --datadir /data \
  --outdir /outputs \
  --logdir /logs \