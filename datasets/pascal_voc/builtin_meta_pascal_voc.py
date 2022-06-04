# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""

# PASCAL VOC categories
PASCAL_VOC_ALL_CATEGORIES = {
    1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor', 'bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor', 'aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor', 'boat', 'cat', 'motorbike', 'sheep', 'sofa'],
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ['bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['boat', 'cat', 'motorbike', 'sheep', 'sofa'],
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'],
    2: ['bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor'],
    3: ['aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor'],
}

def _get_pascal_voc_fewshot_instances_meta():
    ret = {
        "thing_classes": PASCAL_VOC_ALL_CATEGORIES,
        "novel_classes": PASCAL_VOC_NOVEL_CATEGORIES,
        "base_classes": PASCAL_VOC_BASE_CATEGORIES,
    }
    return ret


def _get_builtin_metadata_pascal_voc(dataset_name):
    if dataset_name == "pascal_voc_fewshot":
        return _get_pascal_voc_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
