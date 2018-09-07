#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to preprocess the Potsdam dataset.
#
# Usage:
#   bash ./convert_potsdam.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_potsdam_data.py
#     - convert_potsdam.sh
#     + potsdam
#           + Images
#           + Labels
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./potsdam"


# Root path for POTSDAM dataset.
POTSDAM_ROOT="${WORK_DIR}"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${POTSDAM_ROOT}/Labels"
SEMANTIC_SEG_FOLDER="${POTSDAM_ROOT}/LabelsRaw"

echo "Removing the color map in ground truth annotations..."
python ./remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${POTSDAM_ROOT}/Images"
LIST_FOLDER="${POTSDAM_ROOT}/Index"

echo "Converting POTSDAM dataset..."
python ./build_potsdam_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="png" \
  --output_dir="${OUTPUT_DIR}"
