#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn
import torch

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x,masks=None, targets=None, cls = None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg,reid_loss,contra_loss = self.head(
                fpn_outs,masks, targets, cls, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "reid_loss":reid_loss,
                "contra_loss":contra_loss
            }
            return outputs
            
        else:
            results, mask_output,bg_vector = self.head(fpn_outs)
            if self.head.decode_in_inference==True:
                outputs = {
                    "results": results,
                    "mask_output": mask_output,
                    "bg_vector": bg_vector
                }
                return outputs
            else:
                boxes = results[:, :, :4]

                # Convert [cx, cy, w, h] to [c1x, c1y, c2x, c2y]
                c1x = torch.clamp(boxes[:, :, 0] - boxes[:, :, 2] / 2, min=0, max=x.shape[-1]-1)
                c1y = torch.clamp(boxes[:, :, 1] - boxes[:, :, 3] / 2, min=0, max=x.shape[-2]-1)
                c2x = torch.clamp(boxes[:, :, 0] + boxes[:, :, 2] / 2, min=0, max=x.shape[-1]-1)
                c2y = torch.clamp(boxes[:, :, 1] + boxes[:, :, 3] / 2   , min=0, max=x.shape[-2]-1)             
                # Combine these into the new format
                converted_boxes = torch.stack((c1x, c1y, c2x, c2y), dim=-1)
                
                
                class_conf, class_pred = torch.max(
                    results[:,:, 5 : 5 + 7], 2, keepdim=True
                )
                _, locations = torch.max(
                    results[:,:, 5+7 : 5 + 7+9], 2, keepdim=True
                )
                
                actions = (results[:,:, 5+7+9 : 5 + 7 + 9+19]+0.6)//1

                pred_class =   torch.cat((class_pred, locations, actions), dim=-1)
                
                scores = results[:,:,4:5]*class_conf
                seg_vec = results[:,:,5+7+9+19:]
                
                return converted_boxes,scores, mask_output,bg_vector, pred_class, seg_vec
        
        '''
        else:
            results, mask_output,bg_vector = self.head(fpn_outs)
            if self.head.decode_in_inference==False:
                outputs = {
                    "results": results,
                    "mask_output": mask_output,
                    "bg_vector": bg_vector
                }
                return outputs
            else:
                boxes = results[:, :, :4]

                # Convert [cx, cy, w, h] to [c1x, c1y, c2x, c2y]
                c1x = boxes[:, :, 0] - boxes[:, :, 2] / 2
                c1y = boxes[:, :, 1] - boxes[:, :, 3] / 2
                c2x = boxes[:, :, 0] + boxes[:, :, 2] / 2
                c2y = boxes[:, :, 1] + boxes[:, :, 3] / 2
                
                # Combine these into the new format
                converted_boxes = torch.stack((c1x, c1y, c2x, c2y), dim=-1)

                scores = results[:,:,4:5].expand_as(results[:,:,5:5+7])*results[:,:,5:5+7]
                locations = results[:,:,5+7:5+7+9]
                actions = results[:,:,5+7+9:5+7+9+19]
                seg_vector = results[:,:,5+7+9+19:]
                
                return converted_boxes,scores, mask_output,bg_vector, locations, actions, seg_vector
        '''
