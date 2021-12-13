# import the necessary packages
import numpy as np
import cv2
import os
import re
from os import listdir
from os.path import isfile, join

def order_points(pts):
  # initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype = "float32")
  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis = 1)
  #print(s)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  # return the ordered coordinates
  return rect
def four_point_transform(image, pts):
  # obtain a consistent order of the points and unpack them
  # individually
  rect = order_points(pts)
  #print(rect)
  (tl, tr, br, bl) = rect
  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  # return the warped image
  return warped

def crop2(image,x,name,dem,output_det_path,crop_path):
  #print(path)
  pts = np.array([[x[0],x[1]],[x[2],x[3]],[x[4],x[5]],[x[6],x[7]]])
  # apply the four point tranform to obtain a "birds eye view" of
  # the image
  warped = four_point_transform(image, pts)
  #print(image)
  try:
    cv2.imwrite(crop_path + name + "_" + str(dem).zfill(5) + ".jpg", warped)
    print(crop_path + name + "_" + str(dem).zfill(5) + ".jpg")
    return True
    #print("/content/vin_data/ramdisk/max/90kDICT32px/" + name[:-4] + '__' + str(dem) + '.png ')
  except:
    print(name + " - fail")
    return False

# Crop Images From Detect Files
def crop(rootImage,output_det_path, crop_path):
  files = [f for f in listdir(output_det_path) if isfile(join(output_det_path, f))]
  for f in files:
    print(f)
    fi = open(output_det_path + f,"r")
    lines = fi.readlines()
    dem = 0
    for line in lines:
      dem = dem + 1
      datas = line.split(",")
      image = cv2.imread(rootImage + f.replace('.txt',''))
      coords = [int(x) for x in datas[:8]]
      crop2(image,coords,f.replace('.jpg.txt',''),dem,output_det_path,crop_path)