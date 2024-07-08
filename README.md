# SDPT: Synchronous Dual Prompt Tuning for Fusion-based Visual-Language Pre-trained Models

## Overview

This repository contains the code implementation of the ECCV 2024 paper "SDPT: Synchronous Dual Prompt Tuning for Fusion-based Visual-Language Pre-trained Models." The code is developed based on the GLIP framework, aiming to enhance the performance of fusion-based visual-language pre-trained models through the proposed SDPT method.

## Installation

Since this code is built upon the GLIP framework, the installation process can refer to the [official installation guide of GLIP](https://github.com/microsoft/GLIP) .

Brief steps typically include:
1. Clone the GLIP repository (if not already installed).
2. Set up the environment according to the GLIP README, including installing necessary dependencies.
3. Clone this repository locally and integrate it with the GLIP directory structure (if necessary).

## Datasets

To run and reproduce the experiments in the paper, you need to download the following datasets and place them in the `DATASET` folder:
- COCO (Common Objects in Context)
- LVIS (Large Vocabulary Instance Segmentation)
- ODinW35 (Object Detection in the Wild with 35 Categories)

Please note that these datasets may have copyright and licensing requirements. Ensure you have the right to use them and comply with the corresponding terms and conditions.

## Usage Instructions

### Training SDPT on COCO

To train SDPT using the COCO dataset, you can use the following command with specific configuration options:

```bash
python train.py \
    --config-file configs/pretrain/glip_Swin_L.yaml \
    --restart True \
    --use-tensorboard \
    --override_output_dir OUTPUT_TRAIN_fanew \
    MODEL.BACKBONE.FREEZE_CONV_BODY_AT 1 \
    SOLVER.IMS_PER_BATCH 1 \
    SOLVER.USE_AMP True \
    SOLVER.MAX_ITER 2000 \
    TEST.DURING_TRAINING True \
    TEST.IMS_PER_BATCH 1 \
    SOLVER.FIND_UNUSED_PARAMETERS False \
    SOLVER.BASE_LR 0.00001 \
    SOLVER.LANG_LR 0.00001 \
    DATASETS.DISABLE_SHUFFLE True \
    MODEL.DYHEAD.SCORE_AGG "MEAN" \
    TEST.EVAL_TASK detection \
    AUGMENT.MULT_MIN_SIZE_TRAIN (800,) \
    SOLVER.CHECKPOINT_PERIOD 100 \
    vpt_only -4 \
    froze_fuse 1 \
    MODEL.PROMPT.NUM_TOKENS 700
```

Note that `vpt_only -4`, `froze_fuse 1`, and `MODEL.PROMPT.NUM_TOKENS 700` are SDPT-specific parameters, while the rest are mostly related to the original GLIP framework or runtime details, not directly tied to the SDPT algorithm.

## Contributions and Feedback

We welcome any form of contributions and feedback. If you find any issues or have suggestions for improvement, please report them through GitHub Issues or submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite the ECCV 2024 paper as follows:

```
@inproceedings{your_paper_title,
  title={SDPT: Synchronous Dual Prompt Tuning for Fusion-based Visual-Language Pre-trained Models},
  author={Your Name and Co-Authors},
  booktitle={ECCV 2024},
  year={2024}
}
```
