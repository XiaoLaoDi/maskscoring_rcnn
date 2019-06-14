# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------------------------
# Copyright (C), Data Department of Software Service Center, SiChuan Changhong Electronics Co.Ltd
# ------------------------------------------------------------------------------------------------
# @File          : demo.py
# @Time          : 6/13/19 2:37 PM
# @Author        : X.T.Xiao
# @Email         : xinting.xiao@changhong.com
# @PythonVersion : 
# @Function      : 
# ------------------------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import cv2

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])


coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

# from http://cocodataset.org/#explore?id=345434
# image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
# image = load("http://f.hiphotos.baidu.com/image/pic/item/fc1f4134970a304ed0b72229dbc8a786c8175c48.jpg")
image = load("http://a.hiphotos.baidu.com/image/pic/item/f603918fa0ec08fa3139e00153ee3d6d55fbda5f.jpg")
cv2.namedWindow('ori_img', cv2.WINDOW_NORMAL)
cv2.imshow('ori_img', image)

# compute predictions
predictions = coco_demo.run_on_opencv_image(image)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', predictions)
cv2.imwrite('demo/result.png', predictions)

cv2.waitKey(0)

