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

class Infra_dataset(Dataset):

    def __init__(
        self,
        train=True,
        img_size=(480, 640),
        class_nums = [5,4,3],
        transform=None,
        self_train=False,
        file_root = '/home/etri/Dataset/Infra_BOX/',
        fold = 1):
        
        t_val=''
        
        if train==True:
            t_vals = ['train/','val/']
        else:
            t_vals = ['val/']
        if self_train==True:
            t_val = 'train'
        
        
        self.height = 480
        self.width = 1280 
        self.train = train
        self.class_nums = class_nums

        def is_image(filename):
            return any(filename.endswith(ext) for ext in EXTENSIONS)
        def is_txt(filename):
            return any(filename.endswith(ext) for ext in EXTENSIONS)

        img_fileLabel=[]
        txt_fileLabel=[]
        for t_val in t_vals:
            
            EXTENSIONS = ['.jpg','.png']
            root = file_root+t_val
            img_fileLabel += [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn if is_image(f) and 'img' in dp]

        

            EXTENSIONS = ['.txt']

            txt_fileLabel += [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn if is_txt(f)]
        
        img_fileLabel.sort()
        txt_fileLabel.sort()
        
        if fold>1:
            print("Doing ", fold," fold")
            img_fileLabel=img_fileLabel[::fold]
            txt_fileLabel=txt_fileLabel[::fold]
        
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
        
        
        self.transform = transform # ADDED THIS
        
        
    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, index):
        filename = self.img_list[index]
        filenameGt = self.txt_list[index]
        #print(filename)
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         
        mask = cv2.imread(filename.replace('/img/','/instance/'), cv2.IMREAD_GRAYSCALE)    
        #print(mask.shape)
        
        gtFine = np.loadtxt(filenameGt)
        if gtFine.ndim==1:
            gtFine = np.expand_dims(gtFine, axis=0)
        '''
        if np.any(np.arange(gtFine.shape[0]+1) != np.unique(mask)):
            print(filename,'unique')
            print( gtFine.shape[0],len(np.unique(mask)),np.unique(mask) )
        '''
        #print(gtFine.shape)
        #print(gtFine)
        if self.train==True:
            x_min = np.clip(gtFine[:,0],0,1278)
            x_max = np.clip(gtFine[:,2],1,1279)
            y_min = np.clip(gtFine[:,1],0,478)
            y_max = np.clip(gtFine[:,3],1,479)
            '''
            if np.any(x_min>x_max):
                print(filename,'-x-')
                idx = x_min>x_max
                tmp = x_min[idx]
                x_min[idx]=x_max[idx]
                x_max[idx]=tmp
            if np.any(y_min>y_max):
                print(filename,'-y-')
                idx = y_min>y_max
                tmp = y_min[idx]
                y_min[idx]=y_max[idx]
                y_max[idx]=tmp
            '''
            x = (x_min+x_max)/2.0 /1279
            y = (y_min+y_max)/2.0 /479
            w = (x_max-x_min) / 1279.0 
            h = (y_max-y_min) / 479.0
            all_boxes = np.column_stack([x,y,w,h])
        else:
            x_min = np.clip(gtFine[:,0],0,1278)/1279.0
            x_max = np.clip(gtFine[:,2],1,1279)/1279.0
            y_min = np.clip(gtFine[:,1],0,478)/479.0
            y_max = np.clip(gtFine[:,3],1,479)/479.0
            all_boxes = np.column_stack([x_min,y_min,x_max,y_max])
            
        #print(gtFine)
        '''
        if np.any(gtFine[:,4].astype('int') >= self.class_nums[0]):
            print(filename,'-agent')
            gtFine[:,4] = np.clip(gtFine[:,4],0,self.class_nums[0]-1)
        if np.any(gtFine[:,5].astype('int') >= self.class_nums[1]):
            print(filename,'-loc')
            gtFine[:,5] = np.clip(gtFine[:,5],0,self.class_nums[1]-1)
        '''
        agent_labels = np.eye(self.class_nums[0])[gtFine[:,4].astype('int')]
        loc_labels = np.eye(self.class_nums[1])[gtFine[:,5].astype('int')]
        #print(np.unique(gtFine[:,6]))
        action_labels = gtFine[:,6:].astype('int')
        
        

        #print(all_boxes)
        #print(np.concatenate((agent_labels,loc_labels,action_labels),axis=1))
   
    
        while True:
            transformed1 = self.transform(image=img,mask=mask, bboxes=all_boxes, category_ids=np.concatenate((agent_labels,loc_labels,action_labels),axis=1))
            transformed2 = self.transform(image=img,mask=mask, bboxes=all_boxes, category_ids=np.concatenate((agent_labels,loc_labels,action_labels),axis=1))
            
            if (len(transformed1['bboxes'])==len(transformed2['bboxes'])) == True:
                break
            else:
                print("prob")
        #transformed = self.transform(image=img, bboxes=all_boxes[0], category_ids=agent_labels[0])
        
        height = self.height
        width = self.width
        img_info = (height, width)
        
        
        images = []
        masks = []
        bboxes = []
        targets = []
        img_info_b = []
        index_b = []
        
        images.append(transformed1['image'])
        masks.append(transformed1['mask'])
        bboxes.append(np.array(transformed1['bboxes']))
        targets.append(np.array(transformed1['category_ids']))
        img_info_b.append(img_info)
        index_b.append(index)
        
        images.append(transformed2['image'])
        masks.append(transformed2['mask'])
        bboxes.append(np.array(transformed2['bboxes']))
        targets.append(np.array(transformed2['category_ids']))
        img_info_b.append(img_info)
        index_b.append(index)
        
    
        return images, masks,  bboxes,   targets, img_info_b, index_b
    
def custum_collate(batch):
    
    images = []
    masks = []
    boxes = []
    targets = []
    img_info = []
    index = []
    idx = 0
    
    for sample in batch:
        if idx==0:
            images =sample[0]
            masks = sample[1]
            boxes = sample[2]
            targets= sample[3]
            img_info= sample[4]
            index= sample[5]
            idx+=1
        else:
            images = images+sample[0]
            masks = masks+sample[1]
            boxes = boxes+sample[2]
            targets = targets+sample[3]
            img_info = img_info+sample[4]
            index = index+sample[5]
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
    
    return torch.stack(images, 0), torch.stack(masks, 0), new_boxes, new_targets, img_info, index

