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

def detect(rootImage,output_det_path,config_det,checkpoint_det,threshold_det=0.6):
  listImg = [f for f in listdir(rootImage) if isfile(join(rootImage, f))]
  # initialize the detector
  model = init_detector(config_det, checkpoint_det, device='cuda:0')

  # Detect Scene Text Images
  for name in listImg:
    content = ''

    img = rootImage + name
    # Use the detector to do inference
    result = inference_detector(model, img)

    Boxes = result["boundary_result"]
    for box in Boxes:
      if float(box[8]) >= threshold_det:
        ne = str(int(box[0])) + ',' + str(int(box[1])) + ',' + str(int(box[2])) + ',' + str(int(box[3])) + ',' + str(int(box[4])) + ',' + str(int(box[5])) + ',' + str(int(box[6])) + ',' + str(int(box[7])) + ',###' + '\n'
        content = content + ne

    out_file = open(output_det_path + name + '.txt', 'w')
    out_file.write(content)
