#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from loguru import logger
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, crop_mask
from .lovasz_losses import lovasz_softmax

import math

from .losses import IOUloss,FocalLoss
from .network_blocks import BaseConv, DWConv, CBAM_Block
from torch import Graph, Tensor, Value
     
def decode_outputs(feats: Tensor,
                 strides: Tensor
                 ) -> Tensor:
    #print(feats.shape)
    batch_size = feats[0].shape[0]
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device,
                          dtype=dtype) #+ grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device,
                          dtype=dtype) #+ grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))

    grids_tensor = torch.cat(anchor_points)
    strides_tensor = torch.cat(stride_tensor)
    f_outputs = torch.cat([x.flatten(start_dim=2) for x in feats], dim=2 ).permute(0, 2, 1)
    #print(f_outputs.shape, grids_tensor.shape, strides_tensor.shape)
    box1, box2, box3  = (f_outputs[:,:, :2]+grids_tensor.repeat(batch_size,1,1))*strides_tensor.expand_as(f_outputs[:,:, :2]), torch.exp(f_outputs[:,:, 2:4]) * strides_tensor.expand_as(f_outputs[:,:, 2:4]), f_outputs[:,:, 4:]
    
    #box1 = (f_outputs[:,:, :2] + grids_tensor.expand_as(f_outputs[:,:, :2])) #* strides_tensor.expand_as(f_outputs[:,:, :2])
    #f_outputs[:,:, 2:4] = torch.exp(f_outputs[:,:, 2:4]) * strides_tensor.expand_as(f_outputs[:,:, 2:4])
    
    return torch.cat([box1,box2,box3],2)
'''
def decode_outputs( outputs: Tensor, strides: Tensor)-> Tensor:
    
    grids_tensor = []
    strides_tensor = []
    dtype, device = outputs[0].dtype, outputs[0].device    
    batch_size = outputs.shape[0]
    hw = [torch.Size([60, 160]), torch.Size([30, 80]), torch.Size([15, 40])]
    
    for i, stride in enumerate(strides):
        # Create a grid of coordinates in the feature map.
        h, w = hw[i]
        sx = torch.arange(end=w, device=device, dtype=dtype) 
        sy = torch.arange(end=h, device=device, dtype=dtype) 
        sy, sx = torch.meshgrid(sy, sx)
        grids_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        strides_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))


    # Concatenate all grids and stride tensors.
    grids_tensor = torch.cat(grids_tensor)
    strides_tensor = torch.cat(strides_tensor)

    # Repeat grids and strides for the batch size.
    grids_tensor = grids_tensor.repeat(batch_size,1, 1)
    #strides_tensor = strides.repeat(batch_size, 1, 1)

    # Adjust outputs.
    outputs[..., :2] = (outputs[..., :2] + grids_tensor) * strides_tensor
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides_tensor

    return outputs
'''
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


class YOLOXHead(nn.Module):
    def __init__(
        self,
        class_nums,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.device = torch.device
        self.n_anchors = 1
        self.num_agent =  class_nums[0]
        self.num_loc =  class_nums[1]
        self.num_action  =  class_nums[2]
        self.num_classes = class_nums[0]+class_nums[1]+class_nums[2]
        self.reid_channel = 256
        
        self.decode_in_inference = True  # for deploy, set to False

        self.agent_convs = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.action_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.agent_preds = nn.ModuleList()
        self.action_preds = nn.ModuleList()
        self.loc_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.reid_conv = nn.ModuleList()
        self.reid_feat = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        
        
        self.template =  nn.Sequential(
                    *[
                        
                        Conv(
                            in_channels=self.reid_channel,
                            out_channels=self.reid_channel,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        
                        Conv(
                            in_channels=self.reid_channel,
                            out_channels=self.reid_channel,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        
                        Conv(
                            in_channels=self.reid_channel,
                            out_channels=self.reid_channel,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        
                        nn.Conv2d(
                            in_channels=int(in_channels[0] * width),
                            out_channels=self.reid_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True
                        ),
                        nn.ReLU()
                        
                    ]
                )
        
        
        self.background =  nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(in_channels[2] * width),
                            out_channels=int(in_channels[1] * width),
                            ksize=3,
                            stride=2,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(in_channels[1] * width),
                            out_channels=int(in_channels[0] * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(in_channels[0] * width),
                            out_channels=int(in_channels[0] * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        nn.AdaptiveAvgPool2d(1),
                        
                        nn.Conv2d(
                            in_channels=int(in_channels[0] * width),
                            out_channels=self.reid_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True
                        ),
                        nn.ReLU()
                    ]
                )
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.agent_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.action_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.loc_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reid_conv.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.agent_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_agent,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.action_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_action,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.loc_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_loc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

            self.reid_feat.append(
                nn.Sequential(
                    *[
                        
                        nn.Conv2d(
                            in_channels=int(256 * width),
                            out_channels=self.reid_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True
                        ),
                        nn.ReLU()
                    ]
                )
                
            )
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")#FocalLoss()
        self.iou_loss = IOUloss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction='none')
        self.CEloss = nn.CrossEntropyLoss(ignore_index=0)
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)
       

    
    def tensorrt_compatible_normalize(self, x, dim=1):
        # Assuming ELEMENTWISE operations allow for squaring each element
        squared = x * x
        
        # Assuming REDUCE allows for summing elements across a specified dimension
        sum_squares = squared.sum(dim=dim, keepdim=True)
        
        # Adding epsilon for numerical stability
        sum_squares_eps = sum_squares 
        
        # Assuming ELEMENTWISE operations allow for division
        # No direct square root operation available, so we use division as a way to normalize
        # This is an approximation and not a direct implementation of L2 normalization
        norm = torch.sqrt(sum_squares_eps)
        normalized_x = x / norm
    
        return normalized_x
    
    def initialize_biases(self, prior_prob):
        for conv in self.agent_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.loc_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.action_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin,masks=None, labels=None, agent=None , imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        mask_output = self.tensorrt_compatible_normalize(self.template(xin[0]))
        #mask_output = F.normalize(self.template(xin[0]),p=2,dim=1)
        bg_vector = self.tensorrt_compatible_normalize(self.background(xin[2])).squeeze()
        #bg_vector = F.normalize(self.background(xin[2]),p=2,dim=1).squeeze()
        #print(bg_vector.shape)
        #print("masks: ", masks.shape)
        if self.training:
            resized_masks = torch.nn.functional.interpolate(masks.unsqueeze(1),size=(masks.shape[1]//4,masks.shape[2]//4),
                                                     mode='nearest-exact').squeeze(1)
        
        #print("masks: ", mask_output.shape, resized_masks.shape)
        for k, (agent_conv,loc_conv,action_conv, reg_conv,reid_conv, stride_this_level, x) in enumerate(
            zip(self.agent_convs,self.loc_convs,self.action_convs, self.reg_convs,self.reid_conv, self.strides, xin)
        ):
            #print(x.shape)
            #print('before',x.shape)
            x = self.stems[k](x)
            #print('after',x.shape)
            agent_x = x
            loc_x = x
            action_x = x
            reg_x = x
            reid_x = x

            agent_feat = agent_conv(agent_x)
            agent_output = self.agent_preds[k](agent_feat)

            loc_feat = loc_conv(loc_x)
            loc_output = self.loc_preds[k](loc_feat)

            action_feat = action_conv(action_x)
            action_output = self.action_preds[k](action_feat)

            reid_feat = reid_conv(reid_x)
            reid_output = self.tensorrt_compatible_normalize(self.reid_feat[k](reid_feat))
            #reid_output = F.normalize(self.reid_feat[k](reid_feat),p=2,dim=1)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, agent_output, loc_output, action_output,reid_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), agent_output.sigmoid(), loc_output.sigmoid(), action_output.sigmoid(), reid_output], 1
                )

            outputs.append(output)
        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                agent,
                torch.cat(outputs, 1),
                origin_preds,
                mask_output,
                resized_masks,
                bg_vector,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            #outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2 ).permute(0, 2, 1)
            #print(outputs.shape)
            if self.decode_in_inference:
                return decode_outputs(outputs, self.strides), mask_output,bg_vector
            else:
                return outputs, mask_output,bg_vector

    def get_output_and_grid(self, output, k, stride, dtype):
        
        grid = self.grids[k].to(output.device)

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes + self.reid_channel
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        #print(output,grid,stride)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

 
    
        
    '''
         
    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
        
    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask
    
    '''
    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        agent,
        outputs,
        origin_preds,
        mask_output,
        resized_masks,
        bg_vector,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:5+self.num_classes]  # [batch, n_anchors_all, n_cls]
        reid_preds = outputs[:, :, 5+self.num_classes:]  # [batch, n_anchors_all, n_cls]
        height,width  = imgs.shape[-2:] #img height and widht
        batch_size, _, mask_h, mask_w = mask_output.shape  # batch size, number of masks, mask height, mask width
        
        # calculate targets
        
        label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
        
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        reid_targets =[]
        one_hot_inds=[]
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0
        reid_loss = 0.0
        contra_loss=0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            target_feat = mask_output[batch_idx,:,:]
            bg_feat = bg_vector[batch_idx:batch_idx+1,:]
            target_mask = resized_masks[batch_idx,:,:]
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, :]
                gt_classes = agent[batch_idx, :num_gt, :]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(  # noqa
                    batch_idx,
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    bbox_preds,
                    obj_preds,
                    labels,
                    agent,
                    imgs,
                )
                
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                
                cls_target = gt_matched_classes * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                
                reid_target = reid_preds[batch_idx,fg_mask,:]  #Re-IDs
                one_hot_ind = F.one_hot(matched_gt_inds,num_gt).to(dtype)
                
                merged_feat = reid_target.new_zeros(num_gt, reid_target.shape[1])  #zeros to merge same targets
                count = reid_target.new_zeros(num_gt, 1)  #count to mean

                merged_feat.scatter_add_(0, matched_gt_inds.view(-1, 1).expand(-1, merged_feat.size(1)), reid_target) #make sum
                count.scatter_add_(0, matched_gt_inds.view(-1, 1), reid_target.new_ones(num_fg_img, 1)) #count dumplicates
                involved_idx = (count>0)[:,0] # See found targets
                merged_feat[involved_idx] /= count[involved_idx] # mean , num_gt X reid_feat_num
                #print(merged_feat.shape, bg_feat.shape, gt_bboxes_per_image.shape)
                #print(gt_bboxes_per_image,imgs.shape)
                merged_feat = torch.cat((bg_feat,merged_feat),dim=0) #insert bg_vector
                
                xyxyn = gt_bboxes_per_image / torch.tensor( [width,height,width,height], device=gt_bboxes_per_image.device)
                xyxyn = torch.cat(( torch.tensor([[0.5,0.5,1,1]]).to(xyxyn.device),xyxyn) ,dim=0)
                marea = xyxyn[:, 2:].prod(1)
                xyxyn = xywh2xyxy(xyxyn)
                mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=gt_bboxes_per_image.device)
                
                
                result_mask = torch.nn.functional.relu(torch.matmul(merged_feat,target_feat.view(reid_target.shape[1],-1)).view(num_gt+1,target_feat.shape[1],target_feat.shape[2])) 
                
                
                
                #result_mask = torch.matmul(torch.nn.functional.normalize(merged_feat, dim=1, p=2),\
                 #                          torch.nn.functional.normalize(target_feat, dim=0, p=2).view(reid_target.shape[1],-1)).view(num_gt,target_feat.shape[1],target_feat.shape[2]) # This will be estimated mask
                #result_mask = torch.cat((result_mask.new_zeros(1,result_mask.size(1),result_mask.size(2)),result_mask),dim=0) #pad first layer
                #print(num_gt,num_fg_img)
                                
                #print(result_mask.shape, target_mask.shape)
                
                if torch.max(target_mask)+1<=result_mask.shape[0]:
                    before_loss = self.mse_loss(result_mask,F.one_hot(target_mask.to(torch.long),num_gt+1).permute(2, 0, 1).to(dtype))
                    mask_loss = (crop_mask(before_loss,mxyxy).mean(dim=(1,2))/marea).mean()
                    #mask_loss = lovasz_softmax(result_maskresult_mask,.unsqueeze(0),target_mask.to(torch.long).unsqueeze(0))
                    '''
                    mask_loss = F.cross_entropy(result_mask.unsqueeze(0),target_mask.to(torch.long).unsqueeze(0),weight=loss_weight)\
                    + 0.1* lovasz_softmax(result_mask.unsqueeze(0),target_mask.to(torch.long).unsqueeze(0))
                    '''
                    
                else:
                    print("Error", (torch.unique(target_mask)),result_mask.shape[0])
                    mask_loss =0.0
                reid_loss+=mask_loss
                
                
                ## -------------- Contrastive Learning-----------------
                
                
                
                reid_targets.append(reid_target)
                one_hot_inds.append(one_hot_ind)
                
                
                #conf_matrix = torch.nn.functional.relu(torch.matmul(reid_target,torch.transpose(reid_target, 0, 1)))
                #valid_matrix = torch.matmul(one_hot_ind,torch.transpose(one_hot_ind, 0, 1)) 
                
                
                #contra_loss += self.mse_loss(conf_matrix,valid_matrix).mean()
               
            
            
            
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
                
            if batch_idx%2==1:
                
                reid_targets = torch.cat(reid_targets,0)
                one_hot_inds = torch.cat(one_hot_inds,0)
                
                conf_matrix = torch.nn.functional.relu(torch.matmul(reid_targets,torch.transpose(reid_targets, 0, 1))) #make confusion matrix
                valid_matrix = torch.matmul(one_hot_inds,torch.transpose(one_hot_inds, 0, 1)) 
                
                contra_loss += self.mse_loss(conf_matrix,valid_matrix).mean()
        
                reid_targets =[]
                one_hot_inds=[]

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        #print(cls_preds.view(-1, self.num_classes)[fg_masks].shape)
        #print(cls_targets.shape, cls_preds.view(-1, self.num_classes)[fg_masks].shape)
        #print(cls_targets[0:5,:])
        #print(cls_preds.view(-1, self.num_classes)[fg_masks].shape) = len(fg_masks)*35
        
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + 4*reid_loss + 2*contra_loss

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
            reid_loss,
            contra_loss
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        agent,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask,:]
        gt_classes_ = gt_classes[:,:]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        #print(cls_preds_.shape)
        #print(obj_preds.shape)
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            gt_classes_.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
        #print(cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_().shape)
        #print(obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_().shape)
        cls_preds_ = (
            cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        )
        cls_preds= (cls_preds-0.5)*4
        pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
            '''
        print(
            gt_matched_classes,
            '\n',
            gt_matched_classes.shape,
            '\n-------gt_matched_classes--------------------\n',
            fg_mask,
            '\n',
            fg_mask.shape,
            '\n--------fg_mask-------------------\n',
            pred_ious_this_matching,
            '\n',
            pred_ious_this_matching.shape,
            '\n------pred_ious_this_matching---------------------\n',
            matched_gt_inds,
            '\n',
            matched_gt_inds.shape,
            '\n-----matched_gt_inds----------------------\n',
            num_fg,
            '\n-----End----------------------\n\n\n',
        )
        '''
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = 10
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds,:]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds