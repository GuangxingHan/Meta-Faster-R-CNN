#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""

from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import sys


VOC_classes = ['airplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'dining table', 'dog', 'horse', 'person', 'potted plant', 'sheep',
        'train', 'tv', 'bird', 'bus', 'cow', 'motorcycle', 'couch']
split_dir = 'xxx/cocosplit' # please update the path in your system

for shot in [1, 2, 3, 5, 10, 30]:
    fileids = {}
    for idx, cls in enumerate(VOC_classes):
        json_file = os.path.join(split_dir, 'full_box_{}shot_{}_trainval.json'.format(shot, cls))
        print(json_file)

        coco_api = COCO(json_file)
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        fileids[idx] = list(zip(imgs, anns))

        with open(json_file,'r') as load_f:
            dataset = json.load(load_f)
            save_info = dataset['info']
            save_licenses = dataset['licenses']
            save_images = dataset['images']
            save_categories = dataset['categories']


    combined_imgs = []
    combined_anns = []
    vis_imgs = {}
    for _, fileids_ in fileids.items():
        dicts = []
        for (img_dict, anno_dict_list) in fileids_:
            if img_dict['id'] not in vis_imgs:
                combined_imgs.append(img_dict)
                vis_imgs[img_dict['id']] = True
            combined_anns.extend(anno_dict_list)

    dataset_split = {
        'info': save_info,
        'licenses': save_licenses,
        'images': combined_imgs,
        'annotations': combined_anns,
        'categories': save_categories
    }
    split_file = './new_annotations/final_split_voc_{}_shot_instances_train2014.json'.format(shot)

    with open(split_file, 'w') as f:
        json.dump(dataset_split, f)

