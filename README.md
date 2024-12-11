# evrt-detr

This repository provides the official implementation of "EvRT-DETR: The
Surprising Effectiveness of DETR-based Detection for Event
Cameras" [paper](https://arxiv.org/abs/2412.02890).

EvRT-DETR is a novel object detection model for event cameras that combines the
power of RT-DETR with a LoRA-inspired temporal memory modules. Our approach
achieves state-of-the-art performance on both Gen1 and Gen4/1MPX Prophesee
Automotive Detection datasets while maintaining real-time inference
capabilities.

The instructions below detail how to setup the package and reproduce the
results from our paper.


# Installation & Requirements

The package was tested only under Linux systems.

## Environment

The development environment is based on the
`pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime` Docker container. You can set
up your environment using either Docker or Conda.

### Option 1: Docker Setup

If using Docker (`pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime`), create a
Python virtual environment inside the container to avoid package conflicts:

```bash
python3 -m venv --system-site-packages ~/.venv/evrt-detr
source ~/.venv/evrt-detr/bin/activate
```

### Option 2: Conda Setup

Alternatively, you can create a Conda environment using our provided
configuration:

```bash
conda env create -f contrib/conda_env.yml
conda activate evrt-detr
```

## Requirements

1. COCO evaluation metrics by Prophesee
[psee_adt](https://github.com/realtime-intelligence/psee_adt).
Install the package following the project's instructions.

2. Python dependencies:
```bash
pip install -r requirements.txt
```

## Installation

```bash
pip install -e .
```

For information about default data and output directory paths, refer to
[Default Paths and Environment Variables](#default-paths-and-environment-variables).

NOTE: It is recommended to increase the file descriptor limit before running
the training (see [File Descriptors Limit](#file-descriptors-limit)).
Otherwise, the training is likely to fail when using multiple data workers.

# EvRT-DETR Results Reproduction

This section describes how to reproduce the EvRT-DETR paper results. The
sequence of steps can be summarized as follows:

1. Download and pre-process Prophesee data.
2. [Optional] Download pre-trained models.
3. [Optional] Train models from scratch:
    - Train RT-DETR models on single EBC frames
    - Train EvRT-DETR models on EBC videos
4. Perform model evaluation.


## 1. Dataset Preparation

Due to license restrictions we are unable to distribute pre-processed datasets.
Therefore, the datasets need to be manually downloaded and pre-processed.


### 1.1 Dataset Download

The Gen1 Prophesee dataset can be downloaded from
[this link](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)
(verified 2024-12-10).

The Gen4/1MPX Prophesee dataset can be downloaded from
[this link](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/)
(verified 2024-12-10).


### 1.2 Dataset Pre-processing

To preprocess the datasets into a 2D Stacked Histogram format (to match the
paper), the `scripts/data/psee_to_frames.py` script can be used.

To pre-process the Gen1 dataset, one can use the following command:
```bash
python3 scripts/data/psee_to_frames.py SRC/ DST/ \
    -n 10 --mangling rvt-gen1 -d 50ms --workers N
```

where
  - `SRC/` is a path to the original Prophesee dataset
  - `DST/` is a path where the pre-processed dataset will be stored
  - `-n 10` indicates the number of bins of the stacked histogram
  - `-d 50ms` is a time-window of the stacked histogram
  - `--mangling rvt-gen1` specifies the pre-processing configuration
  - `--workers N` is a number of parallel workers to use for the
     pre-processing. Replace N by the desired number of processes

To pre-process the Gen4/1MPX dataset, one can use a similar command:
```bash
python3 scripts/data/psee_to_frames.py SRC/ DST/ \
    -n 10 --mangling rvt-gen4 -d 50ms --workers N
```
Note however, that Gen4 pre-processing workers can consume quite a lot of RAM
(peak ~20 GB/worker).


NOTE: By default, training scripts expect the datasets to be located under:
  - Gen1: `./data/gen1/gen1_preproc_npz`
  - Gen4: `./data/gen4/gen4_preproc_npz`

You will need to place the pre-processed datasets in these locations for the
training scripts to work.


## 2 [Optional] Downloading Pre-Trained models

The pre-trained models for both datasets are available on
[Zenodo](https://zenodo.org/records/14548751):
- RT-DETR models trained on single EBC frames (Gen1 and Gen4/1MPX)
- EvRT-DETR models trained on EBC videos (Gen1 and Gen4/1MPX)

These pre-trained models are stored in zip archives, named as follows:
 - `DATASET_frame_rtdetr_VARIANT.zip`   -- RT-DETR models
 - `DATASET_video_evrtdetr_VARIANT.zip` -- EvRT-DETR Models

To use the models, download and unpack the relevant zip files manually.

Refer to [Model Directory Structure](#evlearn-model-structure) for details
on the model directory contents.


## 3 Training Models From Scratch

The training of the EvRT-DETR models is staged:
1. At the first stage, simple RT-DETR models are trained on random EBC video
   frames.
2. At the second stage, the EvRT-DETR Memory modules are trained on EBC videos,
   using RT-DETR from stage 1 as an object detection backbone.


### 3.1 Training RT-DETR models

`evrt-detr` provides several scripts to train the RT-DETR models:

```bash
# Gen1 Models
scripts/train/gen1/frame_detection_rtdetr/train_gen1_rtdetr_presnet18.py
scripts/train/gen1/frame_detection_rtdetr/train_gen1_rtdetr_presnet50.py

# Gen4/1MPX Models
scripts/train/gen4/frame_detection_rtdetr/train_gen4_rtdetr_presnet18.py
scripts/train/gen4/frame_detection_rtdetr/train_gen4_rtdetr_presnet50.py
```

Each of these scripts defines a training configuration and calls the training
routine. Feel free to examine and modify those scripts as needed.

To train an RT-DETR PResNet-18 model on the Gen1 dataset, simply run
```bash
python3 scripts/train/gen1/frame_detection_rtdetr/train_gen1_rtdetr_presnet18.py
```

NOTE: It is recommended to increase the file descriptor limit before running
the training (see [File Descriptors Limit](#file-descriptors-limit)).
Otherwise, the training is likely to fail when using multiple data workers.

NOTE: The training might be CPU-bound.
Refer to [Training Performance](#training-performance) for optimization options
if needed.

Once complete, the model will be saved under:
```
outdir/gen1/frame_rtdetr_presnet18/model_m(frame-detection-rtdetr)_default/
```

The output directory can be configured via environment variables (cf.
[Default Paths and Environment Variables](#default-paths-and-environment-variables)
).

Refer to [Model Directory Structure](#evlearn-model-structure) for details
on the directory contents.


### 3.2 Training EvRT-DETR models

The EvRT-DETR models can be trained with the following scripts:

```bash
# Gen1 Models
scripts/train/gen1/video_detection_evrtdetr/train_gen1_evrtdetr_presnet18.py
scripts/train/gen1/video_detection_evrtdetr/train_gen1_evrtdetr_presnet50.py

# Gen4/1MPX Models
scripts/train/gen4/video_detection_evrtdetr/train_gen4_evrtdetr_presnet18.py
scripts/train/gen4/video_detection_evrtdetr/train_gen4_evrtdetr_presnet50.py
```

These scripts expect to find the pre-trained RT-DETR models from the previous
step under `outdir/models`. Please place the pre-trained models there (move
copy entire model directory), or modify the script's `TRANSFER_PATH`
variable to choose another location.

For example, the Gen1 EvRT-DETR PResNet-18 script, expects to find a
pre-trained RT-DETR PResNet-18 model under
```
outdir/models/gen1/frame_rtdetr_presnet18
```

Once the pre-trained RT-DETR model is placed in that location, the EvRT-DETR
training can be started:
```bash
python3 scripts/train/gen1/video_detection_evrtdetr/train_gen1_evrtdetr_presnet18.py
```

NOTE: It is recommended to increase the file descriptor limit before running
the training (see [File Descriptors Limit](#file-descriptors-limit)).
Otherwise, the training is likely to fail when using multiple data workers.

NOTE: The training might be CPU-bound.
Refer to [Training Performance](#training-performance) for optimization options
if needed.

After the training is complete, the trained model will be in:
```
outdir/gen1/video_evrtdetr_presnet18/model_m(vcf-detection-evrtdetr)_default/
```

Refer to [Model Directory Structure](#evlearn-model-structure) for details
on the directory contents.


## 4 Evaluation

To evaluate the COCO mAP metrics `evrt-detr` provides two scripts:
```
scripts/eval_model_frame.py
scripts/eval_model_video.py
```

The `scripts/eval_model_frame.py` can be used to evaluate the RT-DETR model
performance, and the `scripts/eval_model_video.py` can be used to evaluate
the EvRT-DETR model performance.

For example, to evaluate the COCO mAP scores of the manually trained Gen1
RT-DETR model (cf. above) you can use the following command:
```bash
python3 scripts/eval_model_frame.py PATH_TO_MODEL_DIRECTORY
```
where `PATH_TO_MODEL_DIRECTORY` is a path where the trained RT-DETR model is
saved. For the example from section 3.1 it will be:
```
outdir/gen1/frame_rtdetr_presnet18/model_m(frame-detection-rtdetr)_default/
```

When the evaluation is complete, the COCO scores will be printed to the
terminal and saved in the model's `evals/` subdirectory
(cf. [Model Directory Structure](#evlearn-model-structure)).

Similarly, to evaluate the performance of the EvRT-DETR model, one can run
```bash
python3 scripts/eval_model_video.py PATH_TO_MODEL_DIRECTORY --data-name video
```
where `PATH_TO_MODEL_DIRECTORY` is a path where the trained EvRT-DETR model is
saved. For the example from section 3.2 it will be:
```
outdir/gen1/video_evrtdetr_presnet18/model_m(vcf-detection-evrtdetr)_default/
```

# Notes & Reference

## Default Paths and Environment Variables

By default, `evrt-detr` will:
- search for data under the `./data` directory
- save models under the `./outdir` directory

These paths can be changed by setting `EVLEARN_DATA` and `EVLEARN_OUTDIR`
environment variables before running the training/evaluation scripts
(e.g., `export EVLEARN_DATA=/path/to/data/root`).

## evlearn Model Structure

`evlearn` saves each model in a separate directory that contains:
 - `MODEL/config.json` -- model architecture, training, and evaluation
    configurations
 - `MODEL/net_*.pth`  -- PyTorch weights of model networks
 - `MODEL/opt_*.pth`  -- PyTorch weights of training optimizers
 - `MODEL/shed_*.pth` -- PyTorch weights of training schedulers
 - `MODEL/checkpoints/` -- training checkpoints
 - `MODEL/evals/`     -- evaluation results

NOTE: To prevent configuration conflicts, `evlearn` enforces unique
configurations per model directory -- models with different configurations must
be saved in separate directories.


## File Descriptors Limit

Parallel data loading may open too many file descriptors, which can cause
training to fail with errors like:

```
  File "lib/python3.10/multiprocessing/reduction.py", line 164, in recvfds
      raise RuntimeError('received %d items of ancdata' %
              RuntimeError: received 0 items of ancdata
```

If this happens, increase the limit on simultaneously open file descriptors.
For example, on Linux systems:

```bash
ulimit -n 2048
```

This command increases the limit to 2048 file descriptors. You may need to
increase it to higher values if you increase the number of data workers.


## Training Performance

This package and instructions were created with the goals of reproducibility
and simplicity in mind. However, you may find that the default training
scripts are CPU-bound, especially on the Gen4 dataset. There are two reasons
for the CPU-bound training: need to load large amount of data from the disk,
heavy data augmentation pipeline run on a CPU.

If the CPU-bound training is a problem, the most obvious solution is to
increase the number of parallel data loader workers in the script configuration
('workers' parameters).
Make sure to increase the limit of the open file descriptors as well
(see [File Descriptors Limit](#file-descriptors-limit)).

Other solutions are also available. However, they require familiarity with the
deep learning frameworks and we do not have resources to provide support for
them:

1. To reduce IO latency, one can pack the dataset into a compressed HDF5
   format. For example (need to have `hdf5plugin` python package installed):

```bash
python3 scripts/data/pack_npzs_to_hdf5.py \
    --clamp-min frame:0 --clamp-max frame:255 \
    --dtype frame:uint8 -n 16 --chunk-size 1 --compression blosc2 \
    PATH_TO_SOURCE_DATA/SPLIT PATH_TO_DESTINATION/SPLIT
```

To use the compressed dataset:
    - Edit the training script and replace dataset name from 'ebc-video-frame'
      to 'ebc-video-h5frame'.
    - Update `DATA_PATH` variable to point to the new dataset
    - Modify 'label' parameter to any other value (or set to None) to avoid
      config collisions.

2. Data augmentation latency can be reduced by moving data transformations
   from CPU to GPU. This procedure is more involved and requires familiarity
   with the PyTorch framework. The high-level overview of the procedure is:
    - Find the desired model (trainer) in `evlearn/models/` and modify its
      `__init__` and `set_inputs` method.
    - Implement the data augmentation construction found in
      `evlearn/data/data.py` in the `__init__` function.
    - Apply the constructed data augmentations in the `set_inputs` method
      (cf. `evlearn/data/datasets/funcs_frame.py:apply_transforms_to_frame`)


# LICENSE

This project is distributed under the BSD-3 license (see LICENSE file).

This repository includes code from the following projects:

1. `evlearn/bundled/leanbase/`
  - License: BSD-2
  - Purpose: Primitive pytorch routines.
  - The original license text can be found in `evlearn/bundled/leanbase/LICENSE`

2. `evlearn/bundled/rtdetr_pytorch/`
  - Original Project: https://github.com/lyuwenyu/RT-DETR
  - License: Apache-2.0
  - Purpose: Reference RT-DETR implementation.
  - The original license text can be found in `evlearn/bundled/rtdetr_pytorch/LICENSE`

3. `evlearn/bundled/yolox/`
  - Original Project: https://github.com/Megvii-BaseDetection/YOLOX
  - License: Apache-2.0
  - Purpose: Reference YoloX implementation.
  - The original license text can be found in `evlearn/bundled/yolox/LICENSE`

