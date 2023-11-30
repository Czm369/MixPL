# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models.detectors import SemiBaseDetector
from mmdet.structures.bbox import bbox_project
from torch.nn import functional as F
import numpy as np
import math
import os.path as osp


@MODELS.register_module()
class MixPL(SemiBaseDetector):
    """Base class for semi-supervised detectors."""

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.cache_inputs = []
        self.cache_data_samples = []

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        losses.update(**self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
        origin_batch_pseudo_data_samples, batch_info = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher'])
        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_batch_pseudo_data_samples, multi_batch_data_samples['unsup_student'])

        batch_unsup_inputs = copy.deepcopy(multi_batch_inputs['unsup_student'])
        batch_unsup_data_samples = copy.deepcopy(multi_batch_data_samples['unsup_student'])
        batch_unsup_inputs, batch_unsup_data_samples = self.merge(
            *zip(*list(map(self.erase, *self.split(batch_unsup_inputs, batch_unsup_data_samples)))))

        sample_size = len(multi_batch_data_samples['unsup_student'])
        mixup_idxs = np.random.choice(range(sample_size), sample_size, replace=False)
        mosaic_idxs = np.random.choice(range(4), 4, replace=False) + sample_size
        if self.semi_train_cfg.mixup and len(self.cache_inputs) == self.semi_train_cfg.cache_size:
            dst_inputs_list, batch_dst_data_samples = self.split(
                batch_unsup_inputs, batch_unsup_data_samples)
            img_shapes = [tuple(batch_unsup_inputs.shape[-2:])]*batch_unsup_inputs.shape[0]
            src_inputs_list, batch_src_data_samples = self.get_batch(mixup_idxs, img_shapes)

            batch_unsup_inputs, batch_unsup_data_samples = self.merge(*self.mixup(
                dst_inputs_list, batch_dst_data_samples,
                src_inputs_list, batch_src_data_samples))

        if self.semi_train_cfg.mixup:
            losses.update(**rename_loss_dict('mixup_', self.loss_by_pseudo_instances(
                batch_unsup_inputs, batch_unsup_data_samples)))
        else:
            losses.update(**self.loss_by_pseudo_instances(
                batch_unsup_inputs, batch_unsup_data_samples))

        if self.semi_train_cfg.mosaic and len(self.cache_inputs) == self.semi_train_cfg.cache_size:
            if len(self.semi_train_cfg.mosaic_shape) == 1:
                img_shapes = [self.semi_train_cfg.mosaic_shape[0]] * 4
            else:
                mosaic_shape = self.semi_train_cfg.mosaic_shape
                mosaic_h = np.random.randint(
                    min(mosaic_shape[0][0], mosaic_shape[1][0]), max(mosaic_shape[0][0], mosaic_shape[1][0]))
                mosaic_w = np.random.randint(
                    min(mosaic_shape[0][1], mosaic_shape[1][1]), max(mosaic_shape[0][1], mosaic_shape[1][1]))
                img_shapes = [(mosaic_h, mosaic_w)] * 4
            src_inputs_list, batch_src_data_samples = self.get_batch(mosaic_idxs, img_shapes)
            mosaic_inputs, mosaic_data_samples = self.mosaic(src_inputs_list, batch_src_data_samples)
            mosaic_losses = self.loss_by_pseudo_instances(mosaic_inputs, mosaic_data_samples)
            losses.update(**rename_loss_dict('mosaic_', reweight_loss_dict(mosaic_losses, self.semi_train_cfg.mosaic_weight)))
        self.update_cache(multi_batch_inputs['unsup_student'], multi_batch_data_samples['unsup_student'])
        return losses

    def merge(self, inputs_list, batch_data_samples):
        batch_size = len(inputs_list)
        h, w = 0, 0
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            h, w = max(h, img_h), max(w, img_w)
        h, w = max(h, math.ceil(h / 32) * 32), max(w, math.ceil(w / 32) * 32)
        batch_inputs = torch.zeros((batch_size, 3, h, w)).to(self.data_preprocessor.device)
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            batch_inputs[i, :, :img_h, :img_w] = inputs_list[i]
            batch_data_samples[i].set_metainfo({'batch_input_shape': (h, w)})
            batch_data_samples[i].set_metainfo({'pad_shape': (h, w)})
        return batch_inputs, batch_data_samples

    def split(self, batch_inputs, batch_data_samples):
        inputs_list = []
        for i in range(len(batch_inputs)):
            inputs = batch_inputs[i]
            data_samples = batch_data_samples[i]
            img_h, img_w = data_samples.img_shape
            inputs_list.append(inputs[..., :img_h, :img_w])
            data_samples.pop('batch_input_shape')
            data_samples.pop('pad_shape')
        return inputs_list, batch_data_samples

    def update_cache(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        inputs_list, batch_data_samples = self.split(batch_inputs, batch_data_samples)
        cache_size = self.semi_train_cfg.cache_size
        self.cache_inputs.extend(inputs_list)
        self.cache_data_samples.extend(batch_data_samples)
        self.cache_inputs = self.cache_inputs[-cache_size:]
        self.cache_data_samples = self.cache_data_samples[-cache_size:]

    def get_cache(self, idx, img_shape):
        inputs = copy.deepcopy(self.cache_inputs[idx])
        data_samples = copy.deepcopy(self.cache_data_samples[idx])
        inputs, data_samples = self.erase(*self.flip(*self.resize(inputs, data_samples, img_shape)))
        return inputs, data_samples

    def get_batch(self, rand_idxs, img_shapes):
        inputs_list, batch_data_samples = [], []
        for i in range(len(rand_idxs)):
            inputs, data_samples = self.get_cache(rand_idxs[i], img_shapes[i])
            inputs_list.append(inputs)
            batch_data_samples.append(data_samples)
        return inputs_list, batch_data_samples

    def resize(self, inputs, data_samples, img_shape):
        scale = min(img_shape[0] / data_samples.img_shape[0], img_shape[1] / data_samples.img_shape[1])
        inputs = F.interpolate(inputs.unsqueeze(0), scale_factor=scale).squeeze(0)
        data_samples.pop('img_shape')
        data_samples.pop('scale_factor')
        img_h, img_w = inputs.shape[-2:]
        data_samples.set_metainfo({'img_shape': (img_h, img_w)})
        ori_h, ori_w = data_samples.ori_shape
        data_samples.set_metainfo({'scale_factor': (img_w / ori_w, img_h / ori_h)})
        hm = data_samples.pop('homography_matrix')
        matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        data_samples.set_metainfo({'homography_matrix': matrix @ hm})
        data_samples.gt_instances.bboxes *= scale
        data_samples.gt_instances.bboxes[:, 0::2].clamp_(0, img_w)
        data_samples.gt_instances.bboxes[:, 1::2].clamp_(0, img_h)
        return inputs, filter_gt_instances([data_samples])[0]

    def flip(self, inputs, data_samples):
        inputs = inputs.flip(-1)
        img_h, img_w = data_samples.img_shape
        hm = data_samples.pop('homography_matrix')
        matrix = np.array([[-1, 0, img_w], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        data_samples.set_metainfo({'homography_matrix': matrix @ hm})
        flip_flag = data_samples.pop('flip')
        if flip_flag is True:
            data_samples.pop('flip_direction')
            data_samples.set_metainfo({'flip': False})
        else:
            data_samples.set_metainfo({'flip': True})
            data_samples.set_metainfo({'flip_direction': 'horizontal'})
        bboxes = copy.deepcopy(data_samples.gt_instances.bboxes)
        data_samples.gt_instances.bboxes[:, 2] = img_w - bboxes[:, 0]
        data_samples.gt_instances.bboxes[:, 0] = img_w - bboxes[:, 2]
        return inputs, data_samples

    def erase(self, inputs, data_samples):
        def _get_patches(img_shape):
            patches = []
            n_patches = np.random.randint(
                self.semi_train_cfg.erase_patches[0], self.semi_train_cfg.erase_patches[1])
            for _ in range(n_patches):
                ratio = np.random.random() * \
                        (self.semi_train_cfg.erase_ratio[1] - self.semi_train_cfg.erase_ratio[0]) + \
                        self.semi_train_cfg.erase_ratio[0]
                ph, pw = int(img_shape[0] * ratio), int(img_shape[1] * ratio)
                px1 = np.random.randint(0, img_shape[1] - pw)
                py1 = np.random.randint(0, img_shape[0] - ph)
                px2, py2 = px1 + pw, py1 + ph
                patches.append([px1, py1, px2, py2])
            return torch.tensor(patches).to(self.data_preprocessor.device)
        erase_patches = _get_patches(data_samples.img_shape)
        for patch in erase_patches:
            px1, py1, px2, py2 = patch
            inputs[:, py1:py2, px1:px2] = 0
        bboxes = data_samples.gt_instances.bboxes
        left_top = torch.maximum(bboxes[:, None, :2], erase_patches[:, :2])
        right_bottom = torch.minimum(bboxes[:, None, 2:], erase_patches[:, 2:])
        wh = torch.clamp(right_bottom - left_top, 0)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        bboxes_erased_ratio = inter_areas.sum(-1) / (bbox_areas + 1e-7)
        valid_inds = bboxes_erased_ratio < self.semi_train_cfg.erase_thr
        data_samples.gt_instances = data_samples.gt_instances[valid_inds]
        return inputs, data_samples

    def mixup(self, dst_inputs_list, batch_dst_data_samples, src_inputs_list, batch_src_data_samples):
        batch_size = len(dst_inputs_list)
        mixup_inputs_list, batch_mixup_data_samples = [], []
        for i in range(batch_size):
            dst_inputs, dst_data_samples = dst_inputs_list[i], batch_dst_data_samples[i]
            src_inputs, src_data_samples = src_inputs_list[i], batch_src_data_samples[i]
            dst_shape, src_shape = dst_inputs.shape[-2:], src_inputs.shape[-2:]
            mixup_shape = (max(dst_shape[0], src_shape[0]), max(dst_shape[1], src_shape[1]))
            d_x1 = np.random.randint(mixup_shape[1] - dst_shape[1] + 1)
            d_y1 = np.random.randint(mixup_shape[0] - dst_shape[0] + 1)
            d_x2, d_y2 = d_x1 + dst_shape[1], d_y1 + dst_shape[0]
            s_x1 = np.random.randint(mixup_shape[1] - src_shape[1] + 1)
            s_y1 = np.random.randint(mixup_shape[0] - src_shape[0] + 1)
            s_x2, s_y2 = s_x1 + src_shape[1], s_y1 + src_shape[0]
            mixup_inputs = dst_inputs.new_zeros((3, mixup_shape[0], mixup_shape[1]))
            mixup_inputs[:, d_y1:d_y2, d_x1:d_x2] += dst_inputs * 0.5
            mixup_inputs[:, s_y1:s_y2, s_x1:s_x2] += src_inputs * 0.5
            img_meta = dst_data_samples.metainfo
            img_meta['img_shape'] = mixup_shape
            mixup_data_samples = DetDataSample(metainfo=img_meta)
            dst_gt_instances = copy.deepcopy(dst_data_samples.gt_instances)
            dst_gt_instances.bboxes[:, ::2] += d_x1
            dst_gt_instances.bboxes[:, 1::2] += d_y1
            src_gt_instances = copy.deepcopy(src_data_samples.gt_instances)
            src_gt_instances.bboxes[:, ::2] += s_x1
            src_gt_instances.bboxes[:, 1::2] += s_y1
            mixup_data_samples.gt_instances = dst_gt_instances.cat([dst_gt_instances, src_gt_instances])
            mixup_inputs_list.append(mixup_inputs)
            batch_mixup_data_samples.append(mixup_data_samples)
        return mixup_inputs_list, batch_mixup_data_samples

    def mosaic(self, inputs_list, batch_data_samples):
        batch_size = len(inputs_list)
        h, w = 0, 0
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            h, w = max(h, img_h), max(w, img_w)
        h, w = max(h, math.ceil(h / 16) * 16), max(w, math.ceil(w / 16) * 16)
        mosaic_inputs = inputs_list[0].new_zeros((1, 3, h*2, w*2))
        img_meta = batch_data_samples[0].metainfo
        img_meta['batch_input_shape'] = (h*2, w*2)
        img_meta['img_shape'] = (h*2, w*2)
        img_meta['pad_shape'] = (h*2, w*2)
        mosaic_data_samples = [DetDataSample(metainfo=img_meta)]
        mosaic_instances = []
        for i in range(batch_size):
            data_samples_i = copy.deepcopy(batch_data_samples[i])
            gt_instances_i = data_samples_i.gt_instances
            h_i, w_i = data_samples_i.img_shape
            if i == 0:
                mosaic_inputs[0, :, h-h_i:h, w-w_i:w] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w - w_i
                gt_instances_i.bboxes[:, 1::2] += h - h_i
            elif i == 1:
                mosaic_inputs[0, :, h-h_i:h, w:w+w_i] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w
                gt_instances_i.bboxes[:, 1::2] += h - h_i
            elif i == 2:
                mosaic_inputs[0, :, h:h+h_i, w-w_i:w] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w - w_i
                gt_instances_i.bboxes[:, 1::2] += h
            else:
                mosaic_inputs[0, :, h:h+h_i, w:w+w_i] += inputs_list[i]
                gt_instances_i.bboxes[:, ::2] += w
                gt_instances_i.bboxes[:, 1::2] += h
            mosaic_instances.append(gt_instances_i)
        mosaic_data_samples[0].gt_instances = mosaic_instances[0].cat(mosaic_instances)
        return mosaic_inputs, mosaic_data_samples
