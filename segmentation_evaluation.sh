#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Move two-level up to tensorflow/models/research directory.
cd ..

export PYTHONPATH=/home/vignesh/Documents/COS700/Code/models/research:/home/vignesh/Documents/COS700/Code/models/research/slim
echo $PYTHONPATH

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
# Go to datasets folder and download POTSDAM segmentation dataset.
DATASET_DIR="datasets"

# Set up the working directories.
POTSDAM_FOLDER="potsdam"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/${EXP_FOLDER}/export"
POTSDAM_DATASET="${WORK_DIR}/${DATASET_DIR}/${POTSDAM_FOLDER}/tfrecord"

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=75.34%.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${POTSDAM_DATASET}" \
  --max_number_of_evaluations=1