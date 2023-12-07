#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import time
from pycocotools.coco import COCO
import random


# In[4]:


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


# In[14]:


def aug_img_box(img,gtFine):

    aug_imgs = []
    all_boxes = []
    
    x_min = np.clip(gtFine[:,0],0,img.shape[1])
    x_max = np.clip(gtFine[:,2],0,img.shape[1])
    y_min = np.clip(gtFine[:,1],0,img.shape[0])
    y_max = np.clip(gtFine[:,3],0,img.shape[0])

    longer_side = np.maximum(img.shape[1],img.shape[0])
    scale1 = 1000/longer_side
    scale2 = 2000/longer_side
    scale3 = 3000/longer_side
    
    #original    
    aug_imgs.append( cv2.resize(img, dsize=(0, 0), fx=scale1, fy=scale1, interpolation=cv2.INTER_LINEAR))
    all_boxes.append(np.column_stack([scale1*x_min,scale1*y_min,scale1*x_max,scale1*y_max]))
    #larger
    aug_imgs.append( cv2.resize(img, dsize=(0, 0), fx=scale2, fy=scale2, interpolation=cv2.INTER_LINEAR))
    all_boxes.append(np.column_stack([scale2*x_min,scale2*y_min,scale2*x_max,scale2*y_max]))
    #larger
    aug_imgs.append( cv2.resize(img, dsize=(0, 0), fx=scale3, fy=scale3, interpolation=cv2.INTER_LINEAR))
    all_boxes.append(np.column_stack([scale3*x_min,scale3*y_min,scale3*x_max,scale3*y_max]))


    return aug_imgs, all_boxes

def resolve_labels(masks,lable_img):
    from skimage.transform import resize
    
    num_masks = gather_label[0].float().cpu().numpy()[:,0,:,:].shape[0]
    for ii in range(len(gather_label)) :
        if ii==0:#original    
            new_label = resize(gather_label[ii].float().cpu().numpy()[:,0,:,:], (num_masks, lable_img.shape[0], lable_img.shape[1]), order=0, anti_aliasing=False)
            target_label = new_label                   
        elif ii==1: #smaller
            new_label = resize(gather_label[ii].float().cpu().numpy()[:,0,:,:], (num_masks, lable_img.shape[0], lable_img.shape[1]), order=0, anti_aliasing=False)
            target_label += new_label
        elif ii==2: #larger
            new_label = resize(gather_label[ii].float().cpu().numpy()[:,0,:,:], (num_masks, lable_img.shape[0], lable_img.shape[1]), order=0, anti_aliasing=False)
            target_label += new_label
       
    #print(np.max(target_label))
    return target_label


# In[6]:


cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in range(20)]
colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, a in colors]


# In[7]:


from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"#"sam_vit_b_01ec64.pth"#
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


# In[8]:


json_file = '/home/etri/ByteTrack/datasets/mix_mot_ch/annotations/train.json'
coco = COCO(json_file)
ids = coco.getImgIds()


# In[15]:


ii=0
num_files = len(ids)
progress=0
for ids_ in ids:
    ii+=1
    if ii%1==0:
        anno = coco.loadImgs(int(ids_))
        anno_ids = coco.getAnnIds(imgIds=[ids_], iscrowd=False)
        annotations = coco.loadAnns(anno_ids)
        img_file = '/home/etri/ByteTrack/datasets/mix_mot_ch/'+anno[0]['file_name']
        save_path = img_file.replace('img1','instance').replace('.jpg','.png').replace('crowdhuman_','crowdhuman_instance_')
        img = cv2.imread(img_file)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        draw_img = img
        
        lable_img = np.zeros((img.shape[0],img.shape[1],1))
        #print(img.shape,lable_img.shape)
        #print(save_path)
        width = img.shape[1]
        height = img.shape[0]
        
        gtFine=[]
        for annots in annotations:
 
            x1 = np.clip(annots["bbox"][0],0,width-1)
            y1 = np.clip(annots["bbox"][1],0,height-1)
            x2 = np.clip(x1 + annots["bbox"][2],1,width)
            y2 = np.clip(y1 + annots["bbox"][3],1,height)
            
            xyxy = [x1,y1,x2,y2]
            if annots['area']>0 and xyxy[0]<=xyxy[2] and xyxy[1] <=xyxy[3]:
                gtFine.append(xyxy)
        gtFine = np.array(gtFine)
        
        if gtFine.ndim==1:
            gtFine = np.expand_dims(gtFine, axis=0)
            
        imgs, all_boxeses = aug_img_box(img,gtFine)
        gather_label= []
        for (img, all_boxes) in zip(imgs, all_boxeses):
            
            #all_boxes = np.column_stack([x_min*1800/1280,y_min*1200/480,x_max*1800/1280,y_max*1200/480]).astype('int')
            input_boxes = torch.tensor(all_boxes, device=predictor.device)
            #print(input_boxes)
            #print(all_boxes.shape)

            predictor.set_image(img)

            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            #print(masks.shape)
            #masks.cpu().numpy()

            gather_label.append(masks)
            
            
           
        resolve_img = resolve_labels(gather_label,lable_img)
        
        #tmp_img =  np.ones((1,resolve_img.shape[1],resolve_img.shape[2]))*1.5
        tmp_img =  np.ones((1,resolve_img.shape[1],resolve_img.shape[2]))*1.5
        AA = np.concatenate((tmp_img, resolve_img),axis=0)
        BB = np.argmax(AA,axis=0)
        
        '''
        tmp_img =  np.zeros((draw_img.shape[0],draw_img.shape[1],3))
        for ii in range(resolve_img.shape[0]):
            #lable_img[resolve_img[ii,:,:]==True] = ii+1
            tmp_img[BB==ii+1] =colors[ii%len(colors)]
        tmp_img = draw_img*0.7+tmp_img*0.3
        '''    
        folder_path = os.path.dirname(save_path)
        
        
        
        '''
        plt.figure(figsize=(10, 10))
        plt.imshow(tmp_img.astype(np.int64))
        plt.show()
        
        '''
        # Check if the folder exists, and if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cv2.imwrite(save_path, lable_img)
        
        
    progress += 1
    print("\rProgress: {:>3} %".format(progress * 100 / num_files), end=' ')
    sys.stdout.flush()


# In[ ]:


json_file = '/home/etri/ByteTrack/datasets/mix_mot_ch/annotations/val_half.json'
coco = COCO(json_file)
ids = coco.getImgIds()


# In[ ]:


ii=0
num_files = len(ids)
progress=0
for ids_ in ids:
    ii+=1
    if ii%1==0:
        anno = coco.loadImgs(int(ids_))
        anno_ids = coco.getAnnIds(imgIds=[ids_], iscrowd=False)
        annotations = coco.loadAnns(anno_ids)
        img_file = '/home/etri/ByteTrack/datasets/mix_mot_ch/mot_train'+anno[0]['file_name']
        save_path = img_file.replace('img1','instance').replace('.jpg','.png').replace('crowdhuman_','crowdhuman_instance_')
        img = cv2.imread(img_file)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        draw_img = img
        
        lable_img = np.zeros((img.shape[0],img.shape[1],1))
        #print(img.shape,lable_img.shape)
        #print(save_path)
        width = img.shape[1]
        height = img.shape[0]
        
        gtFine=[]
        for annots in annotations:
 
            x1 = np.clip(annots["bbox"][0],0,width-1)
            y1 = np.clip(annots["bbox"][1],0,height-1)
            x2 = np.clip(x1 + annots["bbox"][2],1,width)
            y2 = np.clip(y1 + annots["bbox"][3],1,height)
            
            xyxy = [x1,y1,x2,y2]
            if annots['area']>0 and xyxy[0]<=xyxy[2] and xyxy[1] <=xyxy[3]:
                gtFine.append(xyxy)
        gtFine = np.array(gtFine)
        
        if gtFine.ndim==1:
            gtFine = np.expand_dims(gtFine, axis=0)
            
        imgs, all_boxeses = aug_img_box(img,gtFine)
        gather_label= []
        for (img, all_boxes) in zip(imgs, all_boxeses):
            
            #all_boxes = np.column_stack([x_min*1800/1280,y_min*1200/480,x_max*1800/1280,y_max*1200/480]).astype('int')
            input_boxes = torch.tensor(all_boxes, device=predictor.device)
            #print(input_boxes)
            #print(all_boxes.shape)

            predictor.set_image(img)

            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            #print(masks.shape)
            #masks.cpu().numpy()

            gather_label.append(masks)
            
            
           
        resolve_img = resolve_labels(gather_label,lable_img)
        
        #tmp_img =  np.ones((1,resolve_img.shape[1],resolve_img.shape[2]))*1.5
        tmp_img =  np.ones((1,resolve_img.shape[1],resolve_img.shape[2]))*1.5
        AA = np.concatenate((tmp_img, resolve_img),axis=0)
        BB = np.argmax(AA,axis=0)
        
        '''
        tmp_img =  np.zeros((draw_img.shape[0],draw_img.shape[1],3))
        for ii in range(resolve_img.shape[0]):
            #lable_img[resolve_img[ii,:,:]==True] = ii+1
            tmp_img[BB==ii+1] =colors[ii%len(colors)]
        '''    
        folder_path = os.path.dirname(save_path)
        
        
        #tmp_img = draw_img*0.7+tmp_img*0.3
        
        '''
        plt.figure(figsize=(10, 10))
        plt.imshow(tmp_img.astype(np.int64))
        plt.show()
        
        '''
        # Check if the folder exists, and if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cv2.imwrite(save_path, lable_img)
        
        
    progress += 1
    print("\rProgress: {:>3} %".format(progress * 100 / num_files), end=' ')
    sys.stdout.flush()


# In[ ]:





# In[ ]:





# In[ ]:




