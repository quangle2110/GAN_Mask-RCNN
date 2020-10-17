#!/usr/bin/env python
# coding: utf-8

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import torch.nn as nn
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.modeling.roi_heads.box_head.box_head import ROIBoxHead
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import MaskRCNNLossComputation
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.poolers import Pooler
from PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.rpn.rpn import RPNModule


import datetime
import logging
import time

import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from maskrcnn_benchmark.utils.comm import is_main_process

from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rpn.utils import concat_box_prediction_layers

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import cv2

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

class TensorboardLogger(MetricLogger):
    def __init__(self,
                 log_dir,
                 start_iter=0,
                 delimiter='\t'):

        super(TensorboardLogger, self).__init__(delimiter)
        self.iteration = start_iter
        self.writer = self._get_tensorboard_writer(log_dir)

    @staticmethod
    def _get_tensorboard_writer(log_dir):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                'To use tensorboard please install tensorboardX '
                '[ pip install tensorflow tensorboardX ].'
            )

        if is_main_process():
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
            tb_logger = SummaryWriter('{}-{}'.format(log_dir, timestamp))
            return tb_logger
        else:
            return None

    def update(self, ** kwargs):
        super(TensorboardLogger, self).update(**kwargs)
        if self.writer:
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                assert isinstance(v, (float, int))
                self.writer.add_scalar(k, v, self.iteration)
            self.iteration += 1

def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class NewMaskLossComputaion(MaskRCNNLossComputation):
    def __init__(self, proposal_matcher, discretization_size):
        super(NewMaskLossComputaion, self).__init__(proposal_matcher, discretization_size)
        self.mask_dict = dict(masks=None, targets=None)
        
    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        self.positive_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)
            self.positive_proposals.append(positive_proposals)

        return labels, masks

        
    def __call__(self, proposals, mask_logits, targets):
        labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        
        self.mask_dict.update(masks=mask_logits[positive_inds, labels_pos])
        self.mask_dict.update(targets=mask_targets)
        
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        return mask_loss
        
class AdaptivePooler(Pooler):
    def __init__(self, output_size, scales, sampling_ratio):
        super(AdaptivePooler, self).__init__(output_size, scales, sampling_ratio)
        
    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        
        result = torch.zeros(
                (num_rois, num_channels, output_size, output_size),
                dtype=dtype,
                device=device,
            )
            
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            if level == 0:
                result = pooler(per_level_feature, rois).to(dtype).to(device)
            else:
                result = torch.max(result, pooler(per_level_feature, rois).to(dtype).to(device))
          
        return result
    
class PrPooler(AdaptivePooler):
    def __init__(self, output_size, scales, sampling_ratio):
        super(PrPooler, self).__init__(output_size, scales, sampling_ratio)
        poolers = []
        for scale in scales:
            poolers.append(
                PrRoIPool2D(
                    output_size[0], output_size[1], scale 
                )
            )
        self.poolers = nn.ModuleList(poolers)

class NewROIBoxHead(ROIBoxHead):
    def __init__(self, cfg, in_channels):
        super(NewROIBoxHead, self).__init__(cfg, in_channels)
        self.bbox_dict = dict(bbox=None, target=None)

        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)
        
    def forward(self, features, proposals, targets=None):
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        
        # save bbox result for rois_gan
        if self.training:
            result, box_result = self.reduced_bbox_result([box_regression], proposals)
            self.bbox_dict.update(bbox=result)
            self.bbox_dict.update(target=box_result)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )
    
    def reduced_bbox_result(self, box_regression, proposals):
        
        box_regression = cat(box_regression, dim=0)
        device = box_regression.device
        
        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]

        map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
        
        image_shapes = [box.size for box in proposals]
        boxes_per_image = [len(box) for box in proposals]
        concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        
        prefix_sum_boxes = [boxes_per_image[0]]
        for box_per_images in boxes_per_image[1:]:
            prefix_sum_boxes.append(box_per_images + prefix_sum_boxes[-1])
            
        reduced_boxes_per_image = [0] * len(prefix_sum_boxes)
        i, j = 0, 0
        while i < len(sampled_pos_inds_subset):
            if sampled_pos_inds_subset[i] < prefix_sum_boxes[j]:
                reduced_boxes_per_image[j] += 1
                i += 1
            else:
                j += 1
                
                    
        proposals = self.box_coder.decode(box_regression[sampled_pos_inds_subset[:, None], map_inds], 
                                     concat_boxes[sampled_pos_inds_subset])
        
        proposals = proposals.split(reduced_boxes_per_image, dim=0)
        
        box_targets = self.box_coder.decode(regression_targets[sampled_pos_inds_subset], 
                                     concat_boxes[sampled_pos_inds_subset])
        
        box_targets = box_targets.split(reduced_boxes_per_image, dim=0)
        
        result = []
        for boxes, image_shape in zip(proposals, image_shapes):
            boxlist = BoxList(boxes, image_shape, mode="xyxy")
            boxlist = boxlist.clip_to_image(remove_empty=False)
            result.append(boxlist)
            
        box_result = []
        for boxes, image_shape in zip(box_targets, image_shapes):
            boxlist = BoxList(boxes, image_shape, mode="xyxy")
            boxlist = boxlist.clip_to_image(remove_empty=False)
            box_result.append(boxlist)
        
        return result, box_result

class GAN_LossComputation(object):
    def __init__(self, Dnet, m_coeff=0.5, b_coeff=0.025):
        self.Dnet = Dnet
        self.m_coeff = m_coeff
        self.b_coeff = b_coeff
        self.criterion = nn.BCEWithLogitsLoss()

        self.pooler = PrPooler(
            output_size=(28, 28),
            scales=(0.25, 0.125, 0.0625, 0.03125),
            sampling_ratio=2,
        )
        
        
    def prepare_rois_bbox(self, results, targets, images):
        rois_fake = self.pooler((images,), results)
        rois_real = self.pooler((images,), targets)
        
        return rois_fake, rois_real
    
        
    def __call__(self, mask_in, bbox_in):
        mask_fake, mask_real, positive_proposals, features = mask_in
        results, targets, images = bbox_in
        
        mask_feat = self.pooler(features, positive_proposals).detach()

        detach_features = tuple(feat.detach() for feat in features)
        
        mask_fake, mask_real = mask_fake*mask_feat, mask_real*mask_feat
        bbox_fake, bbox_real = self.prepare_rois_bbox(results, targets, images)
        
        ### D loss
        device = bbox_fake.device
        
        # train with real
        out_mask_reals, out_bbox_reals = self.Dnet(mask_real.detach(), bbox_real.detach())
                
        # train with fake
        out_mask_fakes, out_bbox_fakes = self.Dnet(mask_fake.detach(), bbox_fake.detach())    
                
        d_loss_mask = -torch.mean(torch.abs(out_mask_reals - out_mask_fakes))
                
        d_loss_bbox = self.criterion(out_bbox_reals, torch.full((out_bbox_reals.shape[0],), 1., device=device)) + \
                      self.criterion(out_bbox_fakes, torch.full((out_bbox_fakes.shape[0],), 0., device=device))

        d_losses = dict(d_loss_mask=self.m_coeff*d_loss_mask, d_loss_bbox=self.b_coeff*d_loss_bbox)
                      
        ### G Loss
        # train with real
        out_mask_reals, out_bbox_reals = self.Dnet(mask_real, bbox_real)
                
        # train with fake
        out_mask_fakes, out_bbox_fakes = self.Dnet(mask_fake, bbox_fake)
        
        g_loss_mask = torch.mean(torch.abs(out_mask_reals - out_mask_fakes))
        
        g_loss_bbox = self.criterion(out_bbox_fakes, torch.full((out_bbox_fakes.shape[0],), 1., device=device))
                
        g_losses = dict(g_loss_mask=self.m_coeff*g_loss_mask, g_loss_bbox=self.b_coeff*g_loss_bbox)
            
        
        return g_losses, d_losses

class Mask_RCNN(GeneralizedRCNN):
    def __init__(self, cfg, BoxDnet=None):
        super(Mask_RCNN, self).__init__(cfg)
        
        matcher = Matcher(
                        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
                        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
                        allow_low_quality_matches=False)
        
        self.roi_heads.mask.loss_evaluator = NewMaskLossComputaion(matcher, 
                                                                   cfg.MODEL.ROI_MASK_HEAD.RESOLUTION)
        
        self.roi_heads.mask.feature_extractor.pooler = AdaptivePooler(
                                                        output_size=(24, 24),
                                                        scales=(0.25, 0.125, 0.0625, 0.03125),
                                                        sampling_ratio=2,
                                                    )
        
        self.roi_heads.box = NewROIBoxHead(cfg, self.backbone.out_channels)
        
        self.roi_heads.box.feature_extractor.pooler = AdaptivePooler(
                                                        output_size=(7, 7),
                                                        scales=(0.25, 0.125, 0.0625, 0.03125),
                                                        sampling_ratio=2,
                                                    )
            
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            #Save current features
            self.features = features
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            return losses

        return result

    def reduce_proposals(self, boxlists):
        num_images = len(boxlists)
        for i in range(num_images):
            objectness = boxlists[i].get_field("objectness")
            post_nms_top_n = len(objectness)//2
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = 1
            boxlists[i] = boxlists[i][inds_mask]
        return boxlists

class GAN_RCNN(nn.Module):
    def __init__(self, Gnet, Dnet):
        super(GAN_RCNN, self).__init__()
        self.Gnet = Gnet
        self.Dnet = Dnet
        self.gan_loss_evaluator = GAN_LossComputation(self.Dnet)
        
    def forward(self, images, targets=None):
        outputs = self.Gnet(images, targets)
        
        if self.training:
            losses = outputs
            
            images = to_image_list(images)
            
            g_losses, d_losses = self.gan_loss_evaluator((torch.sigmoid(self.Gnet.roi_heads.mask.loss_evaluator.mask_dict['masks'].unsqueeze(1)),  \
                                                          self.Gnet.roi_heads.mask.loss_evaluator.mask_dict['targets'].unsqueeze(1), \
                                                          self.Gnet.roi_heads.mask.loss_evaluator.positive_proposals, self.Gnet.features), \
                                                          (self.Gnet.roi_heads.box.bbox_dict['bbox'], self.Gnet.roi_heads.box.bbox_dict['target'], \
                                                          images.tensors))
            

            losses.update(g_losses)
                        
            return losses, d_losses
        
        return outputs

class MaskDiscriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(MaskDiscriminator, self).__init__()
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14                       
        )
        
        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 14 x 14
            nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
        )
       
        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
        )
        
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
        )
        
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, ndf * 16, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 1 x 1
        )
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
        
    def forward(self, inputs):
        batchsize = inputs.shape[0]
        out1 = self.convblock1(inputs)
        out2 = self.convblock2(out1)
        out3 = self.convblock3(out2)
        out4 = self.convblock4(out3)
        out5 = self.convblock5(out4)
        output = torch.cat((inputs.view(batchsize,-1),1*out1.view(batchsize,-1),
                            2*out2.view(batchsize,-1),2*out3.view(batchsize,-1),
                            2*out4.view(batchsize,-1),4*out5.view(batchsize,-1)), 1)
        
        return output.view(-1, 1).squeeze(1)
    
class BoxDiscriminator(nn.Module):
    def __init__(self, nc=256, ndf=64):
        super(BoxDiscriminator, self).__init__()
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
        )
        
        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 14 x 14
            nn.Conv2d(ndf * 1, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
        )
       
        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
        )
        
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
        )
        
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            # state size. (1) x 1 x 1
        )
          
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
        
    def forward(self, inputs):
        batchsize = inputs.shape[0]
        out1 = self.convblock1(inputs)
        out2 = self.convblock2(out1)
        out3 = self.convblock3(out2)
        out4 = self.convblock4(out3)
        out5 = self.convblock5(out4)
        
        return out5.view(-1, 1).squeeze(1)
    
class CombinedDiscriminator(nn.Module):
    def __init__(self, MaskDnet, BoxDnet):
        super(CombinedDiscriminator, self).__init__()
        self.MaskDnet = MaskDnet
        self.BoxDnet = BoxDnet
        
    def forward(self, mask_in, bbox_in):
        return self.MaskDnet(mask_in), self.BoxDnet(bbox_in)