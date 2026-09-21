#!/usr/bin/env bash
# Copy this file to env.local.sh, edit DATASET_ROOT, then source it.

export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET_ROOT="/path/to/datasets"
export CV_ROOT="${DATASET_ROOT}/Camera-vehicle1"
export OUTPUT_ROOT="${REPO_ROOT}/outputs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}"

if [ "${DATASET_ROOT}" = "/path/to/datasets" ]; then
    echo "Set DATASET_ROOT in env.local.sh before training." >&2
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "CV_ROOT=${CV_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
