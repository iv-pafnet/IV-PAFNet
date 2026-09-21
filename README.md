# IV-PAFNet

Source-code release for **IV-PAFNet: Infrared-Visible Cross-Modal
Alignment and Fusion Network for Nighttime UAV Object Detection**.

This repository contains the model implementation, paired infrared-visible data
loader, training and evaluation entry points, dataset-conversion utilities,
73-epoch experiment configurations, component variants, and smoke-test utilities.

## 1. Paper scope

The paper reports quantitative experiments on two paired infrared-visible datasets:

- **DroneVehicle**: training, validation, and testing for UAV-view vehicle detection;
- **Camera-vehicle1**: direct cross-scene evaluation of a DroneVehicle-trained model,
  without target-domain fine-tuning.

The 3D-Net samples shown in the architecture figure are illustrations only and are
not an experimental benchmark. LLVIP, FLIR, and M3FD files inherited from upstream
development are not required to reproduce the IV-PAFNet paper results.

## 2. Repository layout

```text
IV-PAFNet/
├── configs/
│   ├── dataset/                     dataset definitions
│   ├── rtdetr/paper_73e/            paper training and ablation configurations
│   └── rtdetr/eval_camera_vehicle1.yml
├── src/
│   ├── core/                        configuration system
│   ├── data/                        paired RGB/IR data loading and transforms
│   ├── solver/                      training and evaluation loops
│   └── zoo/rtdetr/                  IV-PAFNet and RT-DETR components
├── tools/
│   ├── train.py                     training/evaluation entry point
│   ├── smoke_test_release.py        weight-free model construction test
│   ├── make_dronevehicle_smoke_subset.py
│   ├── dronevehicle_voc_to_coco.py annotation conversion
│   └── check_dronevehicle_archive.py
├── environment.yml                  Conda environment specification
├── requirements.txt                 complete pinned Python requirements
├── requirements_no_torch.txt        non-PyTorch requirements
├── env.sh                           path-variable template
├── LICENSE                          IV-PAFNet code license
├── LICENSES/Apache-2.0.txt          RT-DETR/DETR license text
├── LICENSES/BSD-3-Clause.txt        torchvision license text
└── THIRD_PARTY_NOTICES.md           upstream attribution
```

The main model modules are:

- `src/zoo/rtdetr/iv_pafnet.py`: ISA and MSFA;
- `src/zoo/rtdetr/rtdetr.py`: dual-stream `FusionDETR`;
- `src/zoo/rtdetr/hybrid_encoder.py`: multi-scale fusion integration.

## 3. Environment

### 3.1 Recommended environment

- Linux or Windows;
- Python 3.10;
- PyTorch 2.7.1;
- torchvision 0.22.1;
- CUDA-capable NVIDIA GPU.

Create the environment:

```bash
git clone https://github.com/iv-pafnet/IV-PAFNet.git
cd IV-PAFNet

conda env create -f environment.yml
conda activate ivpafnet

# Select the wheel index that matches the installed CUDA driver.
# The example below uses the CUDA 12.8 wheels.
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_no_torch.txt
```

ONNX export is optional and not required for the paper experiments:

```bash
pip install onnx==1.14.0 onnxruntime==1.15.1
```

### 3.2 Environment note

Use Python 3.10 with the pinned PyTorch 2.7.1 / torchvision 0.22.1 pair unless an
alternative environment is recorded with the resulting experiment. On the first training run, the configured
PResNet-50 backbone is downloaded from the upstream RT-DETR release through
`torch.hub`; an internet connection or a populated PyTorch cache is therefore needed.

## 4. Dataset download and preparation

### 4.1 DroneVehicle

- Official project: <https://github.com/VisDrone/DroneVehicle>
- Paper split: 17,990 training pairs, 1,469 validation pairs, and 8,980 test pairs.
- Categories: car, truck, bus, van, and freight car.

Expected directory structure:

```text
${DATASET_ROOT}/DroneVehicle/
├── train/
│   ├── trainimg/
│   ├── trainimgr/
│   ├── trainlabel/
│   └── trainlabelr/
├── val/
│   ├── valimg/
│   ├── valimgr/
│   ├── vallabel/
│   └── vallabelr/
├── test/
│   ├── testimg/
│   ├── testimgr/
│   ├── testlabel/
│   └── testlabelr/
└── coco_annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

The paper converts the original oriented boxes into minimum enclosing horizontal
boxes. If the COCO JSON files are not already available, inspect and run the converter:

```bash
python tools/dronevehicle_voc_to_coco.py --help
```

Large DroneVehicle ZIP files should be checked before extraction:

```bash
python tools/check_dronevehicle_archive.py /path/to/train.zip \
  --manifest damaged_entries.tsv
```

The checker uses Python's ZIP64-aware `zipfile` module and reports unreadable entries.
Obtain the dataset from its authorized source and follow the source repository's
license and terms of use.

### 4.2 Camera-vehicle1

- Dataset page: <https://github.com/iv-pafnet/IV-PAFNet_dataset/releases>
- Dataset index: <https://github.com/iv-pafnet/IV-PAFNet_dataset>
- Size used by the manuscript: 12,483 visible/infrared pairs.

Expected directory structure:

```text
${DATASET_ROOT}/Camera-vehicle1/
├── visible/test/                 12,483 visible images
├── infrared/test/                12,483 infrared images
└── annotations/instances_test.json
```

Only files referenced by `instances_test.json` are evaluated. The DroneVehicle-trained
checkpoint is evaluated directly on Camera-vehicle1 without additional training.

### 4.3 Configure local paths

Copy the environment template and edit the dataset root:

```bash
cp env.sh env.local.sh
# Edit DATASET_ROOT if the default placeholder is still present.
source env.local.sh
```

The resulting variables are:

```bash
DATASET_ROOT=/path/to/datasets
CV_ROOT=${DATASET_ROOT}/Camera-vehicle1
OUTPUT_ROOT=/path/to/IV-PAFNet/outputs
```

## 5. Installation checks

Run the weight-free construction and forward test:

```bash
python tools/smoke_test_release.py
```

For a quicker CPU-only construction check, use:

```bash
python tools/smoke_test_release.py --device cpu --size 160
```

A successful 640 x 640 test reports output shapes equivalent to:

```text
outputs: {'pred_logits': (1, 300, 5), 'pred_boxes': (1, 300, 4)}
```

It also prints total and trainable parameter counts.

To check the complete loading, forward, backward, optimization, validation, and
checkpoint-writing path on a tiny real-data subset:

```bash
python tools/make_dronevehicle_smoke_subset.py \
  --source "${DATASET_ROOT}/DroneVehicle" \
  --output "${OUTPUT_ROOT}/training-smoke-data/DroneVehicle" \
  --train-count 2 --val-count 2

export SMOKE_DATASET_ROOT="${OUTPUT_ROOT}/training-smoke-data"
python tools/train.py -c configs/rtdetr/smoke_train.yml
```

This one-epoch run is an engineering check, not an accuracy experiment.

## 6. Provided 73-epoch configurations

The supplied configurations use 73 completed epochs, AdamW, cosine annealing, EMA, automatic
mixed precision, and a global batch size of 8. The supplied configuration retains
`T_max: 300` and stops after epoch index 72; therefore, “73-epoch protocol” refers to
the stopping point, not a scheduler horizon of 73.

### 6.1 Four-GPU configuration

The four-GPU configuration uses two paired images per GPU:

```bash
torchrun --standalone --nproc_per_node=4 tools/train.py --amp \
  -c configs/rtdetr/paper_73e/DV_Full_S42.yml
```

Module-off comparison configuration with ISA and MSFA disabled:

```bash
torchrun --standalone --nproc_per_node=4 tools/train.py --amp \
  -c configs/rtdetr/paper_73e/DV_Baseline_S42.yml
```

Replace `S42` with `S43` or `S44` to run the supplied seed variants.

### 6.2 Single RTX PRO 6000 adaptation

For a single 96-GB RTX PRO 6000, use the explicitly labeled configuration:

```bash
python tools/train.py --amp \
  -c configs/rtdetr/paper_73e/DV_Full_S42_1xPRO6000.yml

python tools/train.py --amp \
  -c configs/rtdetr/paper_73e/DV_Baseline_S42_1xPRO6000.yml
```

The `S43` and `S44` variants are in the same directory. These configurations retain
the global batch size of 8, but a one-GPU run can differ numerically from the
four-GPU protocol because per-device normalization and execution differ.

### 6.3 Component ablations

The supplied component variants are defined by `B1` through `B6` under
`configs/rtdetr/paper_73e/`. For example:

```bash
torchrun --standalone --nproc_per_node=4 tools/train.py --amp \
  -c configs/rtdetr/paper_73e/B3_FullISA_S42.yml
```

### 6.4 Resume an interrupted run

```bash
python tools/train.py --amp \
  -c configs/rtdetr/paper_73e/DV_Full_S42_1xPRO6000.yml \
  --resume outputs/single_pro6000_73e/DV_Full_S42/checkpoint.pth
```

For a distributed run, use the same `--resume` argument with `torchrun`.
Do not append a new run from epoch 0 to an existing output directory.

## 7. Evaluation

### 7.1 DroneVehicle validation set

```bash
python tools/train.py \
  -c configs/rtdetr/paper_73e/DV_Full_S42.yml \
  --test-only \
  --resume outputs/paper_73e/DV_Full_S42/best.pth
```

Use the configuration matching the checkpoint seed and model variant. The current
evaluation loop does not use autocast, so `--amp` is omitted from test-only commands.

### 7.2 Camera-vehicle1 cross-scene evaluation

```bash
export CV_ROOT="${DATASET_ROOT}/Camera-vehicle1"

python tools/train.py \
  -c configs/rtdetr/eval_camera_vehicle1.yml \
  --test-only \
  --resume /path/to/DroneVehicle-trained-best.pth
```

This command performs evaluation only; it does not fine-tune on Camera-vehicle1.

## 8. Outputs and checkpoint verification

Each training directory contains:

- `log.txt`: one JSON record per completed epoch;
- `checkpoint.pth`: rolling resume checkpoint;
- `best.pth`: checkpoint selected by validation AP;
- numbered periodic checkpoints according to `checkpoint_step`;
- `eval.pth`: COCO evaluation tensor produced by test-only evaluation.

For each reported run, preserve the exact YAML configuration, complete log, selected
checkpoint, and final evaluation output. A completed 73-epoch checkpoint should report
`last_epoch == 72`:

```python
import torch

ckpt = torch.load("best.pth", map_location="cpu", weights_only=False)
print(ckpt["last_epoch"])
print(sorted(ckpt.keys()))
```

The checkpoint should include the model, EMA, optimizer, learning-rate scheduler, and
AMP scaler states when produced by a training run.

## 9. Citation, license, and third-party code

Citation metadata are provided in `CITATION.cff`. License terms are provided in
`LICENSE`. The implementation builds on RT-DETR and related open-source components;
retain the upstream attribution and see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

## 10. Reproducibility checklist

Before archiving a release, verify that it contains or links to:

- this source-code package and its exact version;
- environment and dependency files;
- DroneVehicle and Camera-vehicle1 access and split information;
- preprocessing and annotation-conversion commands;
- configurations for the included seeds and component variants;
- checkpoints, complete logs, and final evaluation outputs;
- data points supporting reported means, standard deviations, tables, and figures;
- an open-source license and third-party notices.

The source archive intentionally excludes datasets, trained weights, and large logs.
Those files should be deposited separately in a public data repository and linked from
the manuscript's Data Availability Statement and the archived code release.
