from mmdet.apis import set_random_seed
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import re
from os import listdir
from os.path import isfile, join
from mmcv import Config
import mmcv
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector
import os.path as osp
from mmocr.apis import init_detector, model_inference


cfg = Config.fromfile('config_path')

# Set up working dir to save files and logs.
cfg.work_dir = 'workdir_path'

# use the mask branch
cfg.load_from = 'pretrained.pth'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 5

# # Change the evaluation metric since we use customized dataset.
# cfg.evaluation.metric = 'mAP'

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 15
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=False)