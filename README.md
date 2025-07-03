# SDPT: Synchronous Dual Prompt Tuning for Fusion-based Visual-Language Pre-trained Models

## Overview

This repository contains the code implementation of the ECCV 2024 paper "SDPT: Synchronous Dual Prompt Tuning for Fusion-based Visual-Language Pre-trained Models." The code is developed based on the GLIP framework, aiming to enhance the performance of fusion-based visual-language pre-trained models through the proposed SDPT method. The paper link is [here](https://arxiv.org/abs/2407.11414).

![detailmethod](https://github.com/user-attachments/assets/19be4d3a-6565-4fb6-b365-38644da8d23f)


## Installation

Since this code is built upon the GLIP framework, the installation process can refer to the [official installation guide of GLIP](https://github.com/microsoft/GLIP) .
But we modified code in GLIP/maskrcnn_benchmark for our SDPT.
Brief steps typically include:
1. clone our repository and open it.
2. create a python environment (we use 3.7.12)
3. python setup.py develop        --this step will install maskrcnn_benchmark according to the ./maskrcnn_benchmark in our repository, and other standard libiaries.Then, you can modify codes in 
 ./maskrcnn_benchmark like us, and explore further.

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
    FROZEE_SWINT True \
    FROZEE_BERT True \
    MODEL.PROMPT.NUM_TOKENS 700
```

Note that `vpt_only -4`, `froze_fuse 1`, and `MODEL.PROMPT.NUM_TOKENS 700` are SDPT-specific parameters, while the rest are mostly related to the original GLIP framework or runtime details, not directly tied to the SDPT algorithm.
All parameters are defined in file maskrcnn_benchmark/config/defaults.py. You can find some explaination or clue about how to use the parameters there. If we did not give detail explaination about some parameter, please global search the parameter name in ./maskrcnn_benchmark for analysis.For example, global search 'froze_fuse', read the context code, and you will find this parameter froze the fuse module in GLIP.
Some important parameter: vpt_only=-4 will add the adapter of our SDPT; MODEL.PROMPT.NUM_TOKENS controls the token length of SDPT.
Change default dataset parameter {DATASETS.TRAIN ('coco_grounding_train',) DATASETS.TEST ('coco_val',)} to apply to ODinW or others. The datasets information should be defined in maskrcnn_benchmark/config/paths_catalog.py. Modify it after you download the datasets.
## Contributions and Feedback

We welcome any form of contributions and feedback. If you find any issues or have suggestions for improvement, please report them through GitHub Issues or submit a Pull Request.

## Citation

If you use this code in your research, please cite the ECCV 2024 paper as follows:

```
@inproceedings{zhou2024sdpt,
  title={SDPT: Synchronous Dual Prompt Tuning for Fusion-Based Visual-Language Pre-trained Models},
  author={Zhou, Yang and Wu, Yongjian and Saiyin, Jiya and Wei, Bingzheng and Lai, Maode and Chang, Eric and Xu, Yan},
  booktitle={European Conference on Computer Vision},
  pages={340--356},
  year={2024},
  organization={Springer}
}
```
