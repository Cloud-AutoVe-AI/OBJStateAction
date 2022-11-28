#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch

import cv2
import numpy as np
import torch.utils as tutils
import json, os
import glob
import albumentations as A

from torch.utils.data import Dataset


def is_part_of_subsets(split_ids, SUBSETS):
    #print(SUBSETS)
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    #print(is_it)
    
    return is_it

def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids

class ETRI_dataset(Dataset):

    def __init__(
        self,
        train=True,
        img_size=(480, 640),
        transform=None,  ):
        if train==True:
            t_val = 'train'
        else:
            t_val = 'train'
        
        self.height = 480
        self.width = 1280
        self.train = train
        
        root = '/data/road-dataset/road/ETRI_Dataset/'+t_val

        EXTENSIONS = ['.jpg','.png']
        def is_image(filename):
            return any(filename.endswith(ext) for ext in EXTENSIONS)

        img_fileLabel = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn if is_image(f)]
        
        img_fileLabel.sort()
        
        
        root = '/data/road-dataset/road/ETRI_Dataset/'+t_val

        EXTENSIONS = ['.txt']
        def is_txt(filename):
            return any(filename.endswith(ext) for ext in EXTENSIONS)

        txt_fileLabel = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn if is_txt(f)]
        
        txt_fileLabel.sort()
        
        
        
        if len(img_fileLabel) != len(txt_fileLabel):
            print("Size mismatch!", len(img_fileLabel), len(txt_fileLabel))
            for ii in range(len(img_fileLabel)):
                print(img_fileLabel[ii])
                print(txt_fileLabel[ii])
                print('---')
                            
        else:
            print("dataset length : ", len(img_fileLabel), len(txt_fileLabel))
        
        self.img_list = img_fileLabel
        self.txt_list = txt_fileLabel
        
        
        
        self.MEANS =[0.485, 0.456, 0.406]
        self.STDS = [0.229, 0.224, 0.225]
        
        
        self.transform = A.Compose(
            [
             
             A.RandomSizedBBoxSafeCrop (width=self.width,height=self.height,erosion_rate=0.3,p=0.5),
             A.Perspective(fit_output=True,p=0.5),
             A.RandomSizedBBoxSafeCrop (width=self.width,height=self.height,erosion_rate=0.3,p=0.5),
             A.OneOf([
                 A.Sharpen(p=0.5),
                 A.AdvancedBlur(p=0.5),
             ],p=0.66),
             A.RandomBrightnessContrast(p=0.5),
             A.RandomShadow(p=1),
            
            A.Normalize(mean=self.MEANS,std=self.STDS),
            A.pytorch.transforms.ToTensorV2()],
            bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
        )

        
    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, index):
        filename = self.img_list[index]
        filenameGt = self.txt_list[index]
        #print(filename)
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                  
        gtFine = np.loadtxt(filenameGt)
        if gtFine.ndim==1:
            gtFine = np.expand_dims(gtFine, axis=0)
        #print(gtFine.shape)
        #print(gtFine.ndim)
        if self.train==True:
            x_min = np.clip(gtFine[:,0],0,1279)
            x_max = np.clip(gtFine[:,2],1,1280)
            y_min = np.clip(gtFine[:,1],0,479)
            y_max = np.clip(gtFine[:,3],1,480)

            x = (x_min+x_max)/2.0 /1280
            y = (y_min+y_max)/2.0 /480
            w = (x_max-x_min) / 1280.0 
            h = (y_max-y_min) / 480.0
            all_boxes = np.column_stack([x,y,w,h])
        else:
            x_min = np.clip(gtFine[:,0],0,1279)/1280.0
            x_max = np.clip(gtFine[:,2],1,1280)/1280.0
            y_min = np.clip(gtFine[:,1],0,479)/480.0
            y_max = np.clip(gtFine[:,3],1,480)/480.0
            all_boxes = np.column_stack([x_min,y_min,x_max,y_max])
            
        agent_labels = np.eye(7)[gtFine[:,4].astype('int')]
        loc_labels = np.eye(9)[gtFine[:,5].astype('int')]
        #print(np.unique(gtFine[:,6]))
        action_labels = gtFine[:,6:].astype('int')
        
        

        height = self.height
        width = self.width
        img_info = (height, width)
        #print(all_boxes)
        #print(np.concatenate((agent_labels,loc_labels,action_labels),axis=1))
        
        
        while True:
            transformed1 = self.transform(image=img, bboxes=all_boxes, category_ids=np.concatenate((agent_labels,loc_labels,action_labels),axis=1))
            transformed2 = self.transform(image=img, bboxes=all_boxes, category_ids=np.concatenate((agent_labels,loc_labels,action_labels),axis=1))
            transformed3 = self.transform(image=img, bboxes=all_boxes, category_ids=np.concatenate((agent_labels,loc_labels,action_labels),axis=1))
                
            
            if (len(transformed1['bboxes'])==len(transformed2['bboxes'])==len(transformed3['bboxes'])) == True:
                break
            else:
                print("prob")
        #transformed = self.transform(image=img, bboxes=all_boxes[0], category_ids=agent_labels[0])
        
        images = []
        boxes = []
        targets = []
        img_info_b = []
        index_b = []
        
        
        images.append(transformed1['image'])
        boxes.append(np.array(transformed1['bboxes']))
        targets.append(np.array(transformed1['category_ids']))
        img_info_b.append(img_info)
        index_b.append(index)
        
        images.append(transformed2['image'])
        boxes.append(np.array(transformed2['bboxes']))
        targets.append(np.array(transformed2['category_ids']))
        img_info_b.append(img_info)
        index_b.append(index)
        
        
        images.append(transformed3['image'])
        boxes.append(np.array(transformed3['bboxes']))
        targets.append(np.array(transformed3['category_ids']))
        img_info_b.append(img_info)
        index_b.append(index)
        
        return images,   boxes,   targets, img_info_b, index_b


def custum_collate(batch):
    
    images = []
    boxes = []
    targets = []
    img_info = []
    index = []
    
    
    '''    
    for sample in batch:
        images.append(sample[0])
        boxes.append(sample[1])
        targets.append(sample[2])
        img_info.append(sample[3])
        index.append(sample[4])
    '''
    idx = 0
        
    for sample in batch:
        if idx==0:
            images =sample[0]
            boxes = sample[1]
            targets= sample[2]
            img_info= sample[3]
            index= sample[4]
            idx+=1
        else:
            images = images+sample[0]
            boxes = boxes+sample[1]
            targets = targets+sample[2]
            img_info = img_info+sample[3]
            index = index+sample[4]
            idx+=1
            
    num_classes = 35
        
    counts = []
    max_len = -1
    
    for bs in boxes:
        max_len = max(max_len, bs.shape[0])
        counts.append(bs.shape[0])
    
    
    new_boxes = torch.zeros([len(boxes), max_len, 4])
    new_targets = torch.zeros([len(boxes), max_len, num_classes])
    
    
    for c1, bs in enumerate(boxes):
        if bs.shape[0]==0:
            pass
        else:
            new_boxes[c1, :counts[c1], :] = torch.from_numpy(bs)
            new_targets[c1, :counts[c1], :] = torch.from_numpy(targets[c1])
        
    height =images[0].shape[1]
    width = images[0].shape[2]
    new_boxes[:,:,0]*=width
    new_boxes[:,:,2]*=width
    new_boxes[:,:,1]*=height
    new_boxes[:,:,3]*=height
    
    return torch.stack(images, 0), new_boxes, new_targets, img_info, index

