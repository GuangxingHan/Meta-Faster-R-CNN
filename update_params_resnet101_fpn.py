"""
Created on Thursday, April 14, 2022

@author: Guangxing Han
"""

import torch
import argparse

parser = argparse.ArgumentParser(description='update_params_resnet101_fpn')
parser.add_argument('model_path', help='model path')

args = parser.parse_args()

# model = torch.load("./output/fsod/meta_training_coco_resnet101_stage_3/model_final.pth")
# model = torch.load("./output/fsod/meta_training_pascalvoc_split1_resnet101_stage_3/model_final.pth")
# model = torch.load("./output/fsod/meta_training_pascalvoc_split2_resnet101_stage_3/model_final.pth")
# model = torch.load("./output/fsod/meta_training_pascalvoc_split3_resnet101_stage_3/model_final.pth")
model_path = args.model_path
model = torch.load(model_path)

for layer_ in list(model['model'].keys()):
    if 'backbone' in layer_:
        model['model'][layer_.replace('backbone', 'backbone.bottom_up')] = model['model'][layer_]
        model['model'].pop(layer_)
    if 'res5' in layer_:
        model['model'][layer_.replace('roi_heads', 'backbone.bottom_up')] = model['model'][layer_]
        model['model'].pop(layer_)

save_path = model_path[:-4] + "_fpn.pth"
print("save_path=", save_path)
torch.save(model, save_path)
