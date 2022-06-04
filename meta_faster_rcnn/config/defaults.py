"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""

from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER.HEAD_LR_FACTOR = 1.0

# ---------------------------------------------------------------------------- #
# Few shot setting
# ---------------------------------------------------------------------------- #
_C.INPUT.FS = CN()
_C.INPUT.FS.FEW_SHOT = False
_C.INPUT.FS.SUPPORT_WAY = 2
_C.INPUT.FS.SUPPORT_SHOT = 10
_C.INPUT.FS.SUPPORT_EXCLUDE_QUERY = False

_C.DATASETS.TRAIN_KEEPCLASSES = 'all'
_C.DATASETS.TEST_KEEPCLASSES = ''
_C.DATASETS.TEST_SHOTS = (1,2,3,5,10,30)
_C.DATASETS.SEEDS = 0

_C.MODEL.ROI_BOX_HEAD.GAMMA_NUM = 2
_C.MODEL.FEWX_BASELINE = True
_C.MODEL.WITH_ALIGNMENT = False
