from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector
import os.path as osp
from mmocr.apis import init_detector, model_inference
import os
import re
from os import listdir
from os.path import isfile, join
from mmcv import Config
import mmcv


def clean(output_det_path,predicted_clean):

  files = [f for f in listdir(output_det_path) if isfile(join(output_det_path, f))]
  print(len(files))
  
  for f in files:
    
    print(output_det_path + f)
    content = ''
    t = open(output_det_path + f,'r')
    lines = t.readlines()
    for line in lines:
      if line.find(',###')<0:
        content = content + line

    new_f = open(predicted_clean+f,'w')
    new_f.write(content)

def recognition(crop_path,checkpoint_reg,config_reg,output_det_path,predicted_clean,threshold_reg = 0.8):
  files = [f for f in listdir(crop_path) if isfile(join(crop_path, f))]

  model = init_detector(config_reg, checkpoint_reg, device="cuda:0")
  for i in range(len(files)):
    content = ''
    new = ''

    img = crop_path + files[i]
    result = model_inference(model, img)

    # print(result['text'])
    # print(result['score'])

    label = result['text']

    label = label.replace('<kt>',' ')
    label = label.replace('<KT>',' ')
    label = label.replace('<Kt>',' ')

    if(files[i].find('.jpeg')>0):
      name = files[i][:(files[i].find('_',4))]
      nu = files[i][files[i].find('_',4)+1:files[i].find('.',files[i].find('_',4))]
      f = open(output_det_path + name,'r')
      content = f.read()
      f.close()
      f3 = open(output_det_path + name,'r')
      lines = f3.readlines()
      x = lines[int(nu)-1].split(',')
      if float(result['score']) >= threshold_reg:
        lab = label.upper()
        new = x[0]+','+x[1]+','+x[2]+','+x[3]+','+x[4]+','+x[5]+','+x[6]+','+x[7]+','+lab
        content = content.replace(lines[int(nu)-1],new+"\n")
      f2 = open(output_det_path + name,'w+')
      f2.write(content)
      f2.close()
      f3.close()
    else:
      name = files[i][:(files[i].find('_',4))] + ".jpg.txt"
      nu = files[i][files[i].find('_',4)+1:files[i].find('.',files[i].find('_',4))]
      f = open(output_det_path + name,'r')
      content = f.read()
      #print(content)
      f.close()
      f3 = open(output_det_path+name,'r')
      lines = f3.readlines()
      x = lines[int(nu)-1].split(',')
      if float(result['score']) >= threshold_reg:
        lab = label.upper()
        new = x[0]+','+x[1]+','+x[2]+','+x[3]+','+x[4]+','+x[5]+','+x[6]+','+x[7]+','+lab
        content = content.replace(lines[int(nu)-1],new+"\n")
      f2 = open(output_det_path+name,'w+')
      f2.write(content)
      f2.close()
      f3.close()