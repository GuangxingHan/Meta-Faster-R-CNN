# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Thursday, April 14, 2022

@author: Guangxing Han
"""
import logging
import numpy as np
import torch
from torch import nn

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .fsod_roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F

from .fsod_fast_rcnn import FsodFastRCNNOutputs

import os

import matplotlib.pyplot as plt
import pandas as pd

from detectron2.data.catalog import MetadataCatalog
import detectron2.data.detection_utils as utils
import pickle
import sys

__all__ = ["FsodRCNN"]


@META_ARCH_REGISTRY.register()
class FsodRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.support_way = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot = cfg.INPUT.FS.SUPPORT_SHOT
        self.logger = logging.getLogger(__name__)
        self.support_dir = cfg.OUTPUT_DIR

        self.evaluation_dataset = 'voc'
        self.evaluation_shot = 10
        self.keepclasses = 'all1'
        self.test_seeds = 0

    def init_support_features(self, evaluation_dataset, evaluation_shot, keepclasses, test_seeds):
        self.evaluation_dataset = evaluation_dataset
        self.evaluation_shot = evaluation_shot
        self.keepclasses = keepclasses
        self.test_seeds = test_seeds

        if self.evaluation_dataset == 'voc':
            self.init_model_voc()
        elif self.evaluation_dataset == 'coco':
            self.init_model_coco()

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        images, support_images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            for x in batched_inputs:
                x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))
            
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        features = self.backbone(images.tensor)

        # support branches
        support_bboxes_ls = []
        for item in batched_inputs:
            bboxes = item['support_bboxes']
            for box in bboxes:
                box = Boxes(box[np.newaxis, :])
                support_bboxes_ls.append(box.to(self.device))
        
        B, N, C, H, W = support_images.tensor.shape
        assert N == self.support_way * self.support_shot

        support_images = support_images.tensor.reshape(B*N, C, H, W)
        support_features = self.backbone(support_images)
        
        # support feature roi pooling
        feature_pooled = self.roi_heads.roi_pooling(support_features, support_bboxes_ls)

        support_box_features = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features], support_bboxes_ls)

        detector_loss_cls = []
        detector_loss_box_reg = []
        rpn_loss_rpn_cls = []
        rpn_loss_rpn_loc = []
        for i in range(B): # batch
            # query
            query_gt_instances = [gt_instances[i]] # one query gt instances
            query_images = ImageList.from_tensors([images[i]]) # one query image

            query_feature_res4 = features['res4'][i].unsqueeze(0) # one query feature for attention rpn
            query_features = {'res4': query_feature_res4} # one query feature for rcnn

            # positive support branch ##################################
            pos_begin = i * self.support_shot * self.support_way
            begin_rel = 0
            for idx in range(begin_rel+1, len(batched_inputs[i]['support_cls'])):
                if batched_inputs[i]['support_cls'][idx] != batched_inputs[i]['support_cls'][begin_rel]:
                    break
            pos_end = pos_begin + idx
            pos_support_features = feature_pooled[pos_begin:pos_end].mean(0, True) # pos support features from res4, average all supports, for rcnn
            pos_support_features_pool = pos_support_features.mean(dim=[2, 3], keepdim=True) # average pooling support feature for attention rpn
            pos_correlation = F.conv2d(query_feature_res4, pos_support_features_pool.permute(1,0,2,3), groups=1024) # attention map

            pos_features = {'res4': pos_correlation} # attention map for attention rpn
            pos_support_box_features = support_box_features[pos_begin:pos_end].mean(0, True)
            pos_proposals, pos_anchors, pos_pred_objectness_logits, pos_gt_labels, pos_pred_anchor_deltas, pos_gt_boxes = self.proposal_generator(query_images, pos_features, pos_support_features_pool, query_gt_instances) # attention rpn
            pos_pred_class_logits, pos_pred_proposal_deltas, pos_detector_proposals = self.roi_heads(query_images, query_features, pos_support_box_features, pos_proposals, query_gt_instances) # pos rcnn

            # negative support branch ##################################
            neg_end = pos_end
            for way in range(self.support_way-1):
                if neg_end >= pos_begin + len(batched_inputs[i]['support_cls']):
                    break
                neg_begin = neg_end
                begin_rel = neg_begin - pos_begin
                for idx in range(begin_rel+1, len(batched_inputs[i]['support_cls'])):
                    if batched_inputs[i]['support_cls'][idx] != batched_inputs[i]['support_cls'][begin_rel]:
                        break
                if batched_inputs[i]['support_cls'][idx] != batched_inputs[i]['support_cls'][begin_rel]:
                    neg_end = pos_begin + idx
                else:
                    neg_end = pos_begin + idx + 1

                neg_support_features = feature_pooled[neg_begin:neg_end].mean(0, True)
                neg_support_features_pool = neg_support_features.mean(dim=[2, 3], keepdim=True)
                neg_correlation = F.conv2d(query_feature_res4, neg_support_features_pool.permute(1,0,2,3), groups=1024)

                neg_features = {'res4': neg_correlation}

                neg_support_box_features = support_box_features[neg_begin:neg_end].mean(0, True)
                neg_proposals, neg_anchors, neg_pred_objectness_logits_tmp, neg_gt_labels_tmp, neg_pred_anchor_deltas_tmp, neg_gt_boxes_tmp = self.proposal_generator(query_images, neg_features, neg_support_features_pool, query_gt_instances)
                neg_pred_class_logits_tmp, neg_pred_proposal_deltas_tmp, neg_detector_proposals_tmp = self.roi_heads(query_images, query_features, neg_support_box_features, neg_proposals, query_gt_instances)

                if way == 0:
                    neg_pred_objectness_logits = neg_pred_objectness_logits_tmp
                    neg_gt_labels = neg_gt_labels_tmp
                    neg_pred_anchor_deltas = neg_pred_anchor_deltas_tmp
                    neg_gt_boxes = neg_gt_boxes_tmp
                    neg_pred_class_logits = neg_pred_class_logits_tmp
                    neg_pred_proposal_deltas = neg_pred_proposal_deltas_tmp
                    neg_detector_proposals = neg_detector_proposals_tmp
                else:
                    neg_pred_objectness_logits += neg_pred_objectness_logits_tmp
                    neg_gt_labels += neg_gt_labels_tmp
                    neg_pred_anchor_deltas += neg_pred_anchor_deltas_tmp
                    neg_gt_boxes += neg_gt_boxes_tmp
                    neg_pred_class_logits = torch.cat([neg_pred_class_logits, neg_pred_class_logits_tmp], dim=0)
                    neg_pred_proposal_deltas = torch.cat([neg_pred_proposal_deltas, neg_pred_proposal_deltas_tmp], dim=0)
                    neg_detector_proposals += neg_detector_proposals_tmp

            # rpn loss
            outputs_images = ImageList.from_tensors([images[i], images[i]])

            outputs_pred_objectness_logits = [torch.cat(pos_pred_objectness_logits + neg_pred_objectness_logits, dim=0)]
            outputs_pred_anchor_deltas = [torch.cat(pos_pred_anchor_deltas + neg_pred_anchor_deltas, dim=0)]
            
            outputs_anchors = pos_anchors # + neg_anchors

            # convert 1 in neg_gt_labels to 0
            for item in neg_gt_labels:
                item[item == 1] = 0

            outputs_gt_boxes = pos_gt_boxes + neg_gt_boxes #[None]
            outputs_gt_labels = pos_gt_labels + neg_gt_labels

            if self.training:
                proposal_losses = self.proposal_generator.losses(
                    outputs_anchors, outputs_pred_objectness_logits, outputs_gt_labels, outputs_pred_anchor_deltas, outputs_gt_boxes)
                proposal_losses = {k: v * self.proposal_generator.loss_weight for k, v in proposal_losses.items()}
            else:
                proposal_losses = {}

            # detector loss
            detector_pred_class_logits = torch.cat([pos_pred_class_logits, neg_pred_class_logits], dim=0)
            detector_pred_proposal_deltas = torch.cat([pos_pred_proposal_deltas, neg_pred_proposal_deltas], dim=0)
            for item in neg_detector_proposals:
                item.gt_classes = torch.full_like(item.gt_classes, 1)
            
            #detector_proposals = pos_detector_proposals + neg_detector_proposals
            detector_proposals = [Instances.cat(pos_detector_proposals + neg_detector_proposals)]
            if self.training:
                predictions = detector_pred_class_logits, detector_pred_proposal_deltas
                detector_losses = self.roi_heads.box_predictor.losses(predictions, detector_proposals)

            rpn_loss_rpn_cls.append(proposal_losses['loss_rpn_cls'])
            rpn_loss_rpn_loc.append(proposal_losses['loss_rpn_loc'])
            detector_loss_cls.append(detector_losses['loss_cls'])
            detector_loss_box_reg.append(detector_losses['loss_box_reg'])
        
        proposal_losses = {}
        detector_losses = {}

        proposal_losses['loss_rpn_cls'] = torch.stack(rpn_loss_rpn_cls).mean()
        proposal_losses['loss_rpn_loc'] = torch.stack(rpn_loss_rpn_loc).mean()
        detector_losses['loss_cls'] = torch.stack(detector_loss_cls).mean() 
        detector_losses['loss_box_reg'] = torch.stack(detector_loss_box_reg).mean()


        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def init_model_voc(self):
        if 1:
            if self.test_seeds == 0:
                support_path = './datasets/pascal_voc/voc_2007_trainval_{}_{}shot.pkl'.format(self.keepclasses, self.evaluation_shot)
            elif self.test_seeds >= 0:
                support_path = './datasets/pascal_voc/seed{}/voc_2007_trainval_{}_{}shot.pkl'.format(self.test_seeds, self.keepclasses, self.evaluation_shot)

            support_df = pd.read_pickle(support_path)

            min_shot = self.evaluation_shot
            max_shot = self.evaluation_shot
            self.support_dict = {'res4_avg': {}, 'res5_avg': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join('./datasets/pascal_voc', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                min_shot = min(min_shot, len(support_box_all))
                max_shot = max(max_shot, len(support_box_all))
                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                support_features = self.backbone(support_images.tensor)

                res4_pooled = self.roi_heads.roi_pooling(support_features, support_box_all)
                res4_avg = res4_pooled.mean(0, True)
                res4_avg = res4_avg.mean(dim=[2,3], keepdim=True)
                self.support_dict['res4_avg'][cls] = res4_avg.detach().cpu().data

                res5_feature = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features], support_box_all)
                res5_avg = res5_feature.mean(0, True)
                self.support_dict['res5_avg'][cls] = res5_avg.detach().cpu().data

                del res4_avg
                del res4_pooled
                del support_features
                del res5_feature
                del res5_avg

            
            for res_key, res_dict in self.support_dict.items():
                for cls_key, feature in res_dict.items():
                    self.support_dict[res_key][cls_key] = feature.cuda()

            print("min_shot={}, max_shot={}".format(min_shot, max_shot))


    def init_model_coco(self):
        if 1:
            if self.keepclasses == 'all':
                if self.test_seeds == 0:
                    support_path = './datasets/coco/full_class_{}_shot_support_df.pkl'.format(self.evaluation_shot)
                elif self.test_seeds > 0:
                    support_path = './datasets/coco/seed{}/full_class_{}_shot_support_df.pkl'.format(self.test_seeds, self.evaluation_shot)
            else:
                if self.test_seeds == 0:
                    support_path = './datasets/coco/{}_shot_support_df.pkl'.format(self.evaluation_shot)
                elif self.test_seeds > 0:
                    support_path = './datasets/coco/seed{}/{}_shot_support_df.pkl'.format(self.test_seeds, self.evaluation_shot)

            support_df = pd.read_pickle(support_path)

            metadata = MetadataCatalog.get('coco_2014_train')
            # unmap the category mapping ids for COCO
            reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
            support_df['category_id'] = support_df['category_id'].map(reverse_id_mapper)

            min_shot = self.evaluation_shot
            max_shot = self.evaluation_shot
            self.support_dict = {'res4_avg': {}, 'res5_avg': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join('./datasets/coco', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                min_shot = min(min_shot, len(support_box_all))
                max_shot = max(max_shot, len(support_box_all))
                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                support_features = self.backbone(support_images.tensor)

                res4_pooled = self.roi_heads.roi_pooling(support_features, support_box_all)
                res4_avg = res4_pooled.mean(0, True)
                res4_avg = res4_avg.mean(dim=[2,3], keepdim=True)
                self.support_dict['res4_avg'][cls] = res4_avg.detach().cpu().data

                res5_feature = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features], support_box_all)
                res5_avg = res5_feature.mean(0, True)
                self.support_dict['res5_avg'][cls] = res5_avg.detach().cpu().data

                del res4_avg
                del res4_pooled
                del support_features
                del res5_feature
                del res5_avg

            
            for res_key, res_dict in self.support_dict.items():
                for cls_key, feature in res_dict.items():
                    self.support_dict[res_key][cls_key] = feature.cuda()

            print("min_shot={}, max_shot={}".format(min_shot, max_shot))


    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        B, _, _, _ = features['res4'].shape
        assert B == 1 # only support 1 query image in test
        assert len(images) == 1
        support_proposals_dict = {}
        support_box_features_dict = {}
        proposal_num_dict = {}
 
        for cls_id, res4_avg in self.support_dict['res4_avg'].items():
            query_images = ImageList.from_tensors([images[0]]) # one query image

            query_features_res4 = features['res4'] # one query feature for attention rpn
            query_features = {'res4': query_features_res4} # one query feature for rcnn

            # support branch ##################################
            support_box_features = self.support_dict['res5_avg'][cls_id]

            correlation = F.conv2d(query_features_res4, res4_avg.permute(1,0,2,3), groups=1024) # attention map

            support_correlation = {'res4': correlation} # attention map for attention rpn

            proposals, _ = self.proposal_generator(query_images, support_correlation, res4_avg, None)
            support_proposals_dict[cls_id] = proposals
            support_box_features_dict[cls_id] = support_box_features

            if cls_id not in proposal_num_dict.keys():
                proposal_num_dict[cls_id] = []
            proposal_num_dict[cls_id].append(len(proposals[0]))

            del support_box_features
            del correlation
            del res4_avg
            del query_features_res4

        results, _ = self.roi_heads.eval_with_support(query_images, query_features, support_proposals_dict, support_box_features_dict)
        
        if do_postprocess:
            return FsodRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        if self.training:
            # support images
            support_images = [x['support_images'].to(self.device) for x in batched_inputs]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)

            return images, support_images
        else:
            return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
