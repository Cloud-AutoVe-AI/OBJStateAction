from data import datasets, dataset_backup
import albumentations as A
from data import custum_collate
from data import custom_collate
import numpy as np
import glob, cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import albumentations.pytorch

import torch
import torch.distributed as dist
import torch.nn as nn

import os
import random
import torch.utils.data as data_utils

import pytorch_lightning as pl
from yolox.exp import Exp
import modules.evaluation as evaluate
from pytorch_lightning.plugins import DDPPlugin
import warnings
warnings.filterwarnings("ignore")
from yolox.utils import postprocess
import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def get_individual_labels(gt_boxes, tgt_labels):
    # print(gt_boxes.shape, tgt_labels.shape)
    new_gts = np.zeros((gt_boxes.shape[0]*30, 5))
    ccc = 0
    for n in range(tgt_labels.shape[0]):
        for t in range(tgt_labels.shape[1]):
            if tgt_labels[n,t]>0:
                new_gts[ccc, :4] = gt_boxes[n,:]
                new_gts[ccc, 4] = t
                ccc += 1
    return new_gts[:ccc,:]


def get_model(depth=1, width=1, num_classes=7):
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


    in_channels = [256, 512, 726]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model


def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()

    for name, param in state_dict.items():
        name = name[5:]
        if name not in own_state:
            print('not loaded:', name)
            continue
        else:
            try:
                if 'backbone.backbone' in name:
                    param.requires_grad = False
                    #print('freeze: ', name)
                else:
                    param.requires_grad = True
                    
                own_state[name].copy_(param)
                #print('Loaded : ', name)
            except:
                print('Imcompatible: ', name)
                pass
                
    #for name, param in model.named_parameters():
    #    if 'backbone.backbone' in name:
    #        param.requires_grad=False
    #    print(name, param.requires_grad)
        
    return model

    
  



#MAX_ITER =len(train_data_loader)
class DataModuleRoad(pl.LightningDataModule):
    def __init__(self, ):
        super().__init__()
          
        self.MEANS =[0.485, 0.456, 0.406]
        self.STDS = [0.229, 0.224, 0.225]
        
        self.idx= 1
        
        
        self.base_w_h = [[1280, 480],[768, 480]]
        self.BATCH_SIZE =[6,6,6]      #[48,48,48]    #
        


        
    def train_dataloader(self):
        
        target_idx = 0# (self.idx+1)%2
        
        w_h = self.base_w_h[target_idx]
        BATCH_SIZE  = self.BATCH_SIZE[target_idx]
        # Defining transforms to be applied on the data

        if target_idx==1:
            self.train_transform = A.Compose(
                [
                 A.Crop(x_min=38,y_min=25,x_max=1242,y_max=780),
                 A.ShiftScaleRotate(scale_limit=0.15,rotate_limit=(0,10),p=0.5),
                 #A.RandomBrightnessContrast(p=0.5),
                 A.RandomResizedCrop(width=w_h[0],height=w_h[1],scale=(0.8,1),p=1) ,
                 A.ColorJitter(p=0.5),
                 A.RandomShadow(p=0.5),

                A.Normalize(mean=self.MEANS,std=self.STDS),
                A.pytorch.transforms.ToTensorV2()],
                bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
            )

            training= True
            self.train_dataset = datasets.ROADDataset(train=training,transform=self.train_transform)
        else:
            self.train_transform = A.Compose(
                [
                 A.OneOf([
                     A.RandomSizedBBoxSafeCrop (width=w_h[0],height=w_h[1],p=0.5),
                     A.ShiftScaleRotate(scale_limit=0.15,rotate_limit=(0,7),p=0.5)
                 ],p=1),
                 A.RandomShadow(p=0.5),
                 A.RandomBrightnessContrast(p=1),
                 #A.RandomResizedCrop(width=w_h[0],height=w_h[1],scale=(0.8,1),p=1) ,
                 #A.ColorJitter(p=0.5),

                A.Normalize(mean=self.MEANS,std=self.STDS),
                A.pytorch.transforms.ToTensorV2()],
                bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
            )

            training= True
            self.train_dataset = datasets.ETRI_dataset(train=training,transform=self.train_transform)
        print("Training Set : ", len(self.train_dataset))
        print("Training idx : ",w_h,BATCH_SIZE,self.idx)
        
        train_data_loader = data_utils.DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,collate_fn=custum_collate, pin_memory=True, drop_last=True)

          # Generating train_dataloader
        return train_data_loader
  


    def val_dataloader(self):
        

        w_h = self.base_w_h[1]
        BATCH_SIZE  = self.BATCH_SIZE[1]
                
        self.val_transform = A.Compose(
            [
             #A.Crop(x_min=38,y_min=25,x_max=1242,y_max=780),
             #A.Resize(width=w_h[0],height=w_h[1]),
             #A.ShiftScaleRotate(p=0.5),
             A.Normalize(mean=self.MEANS,std=self.STDS),
            A.pytorch.transforms.ToTensorV2()],
            bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']),
        )
  
        training= False
        self.val_dataset = dataset_backup.ETRI_dataset(train=training,transform=self.val_transform)
        print("Validation Set : ", len(self.val_dataset))
        self.idx+=1
        print("Validation idx : ",self.idx)
        
        val_data_loader = data_utils.DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,collate_fn=custom_collate, pin_memory=True, drop_last=True)
          # Generating val_dataloader
        return val_data_loader
  
  


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.exp = Exp()
        
        model = self.exp.get_model()
        ckpt = torch.load('/data/road-dataset/Track_YOLOX/lightning_logs/default/version_186/checkpoints/last.ckpt', map_location='cpu')
        model = load_my_state_dict(model,ckpt["state_dict"])
        
        
        #ckpt = torch.load('/data/road-dataset/My_YOLOX/yolox_l.pth', map_location='cpu')
        #model = load_my_state_dict(model,ckpt["model"])
        self.base = model
        
        #self.apply(self._init_weights)
        self.all_classes = ['Ped', 'Cyc', 'Mobike', 'Car', 'Bus', 'EmVeh', 'TL']
        #self.f = open('./log.txt', "a")

    def foward(self,images, gt_boxes, gt_targets):
        return self.base(images, gt_boxes, gt_targets)
    
    def configure_optimizers(self):
        #optimizer = self.exp.get_optimizer(BATCH_SIZE)
        #scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img*BATCH_SIZE,MAX_ITER)
        #optimizer = torch.optim.AdamW(self.parameters(),lr=0)
        optimizer = self.exp.get_optimizer(48)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=2, eta_max=1e-3,  T_up=5, gamma=0.5)
        #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=2, eta_max=3e-4,  T_up=5, gamma=0.8)
        #optimizer = self.exp.get_optimizer(48)
        #optimizer = torch.optim.AdamW(self.parameters(),lr=1e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,                                                                T_mult=2, eta_min=0.00001)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,250], gamma=0.2)
        #scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img*BATCH_SIZE,MAX_ITER)
        
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120], gamma=0.5)
        
        
        return [optimizer], [scheduler]
    
    def step(self, batch):
        
        img, box,cls,_,_  = batch
        #print(img.shape)
        #print(box)
        result = self.foward(img, box,cls)
        loss = result['total_loss']
        self.log('iou_loss',result['iou_loss'],logger=True,sync_dist=True)
        self.log('conf_loss',result['conf_loss'],logger=True,sync_dist=True)
        self.log('cls_loss',result['cls_loss'],logger=True,sync_dist=True)
        self.log('loss_agent',result['loss_agent'],logger=True,sync_dist=True)
        self.log('loss_loc',result['loss_loc'],logger=True,sync_dist=True)
        self.log('loss_action',result['loss_action'],logger=True,sync_dist=True)
        self.log('loss_reid',result['loss_reid'],logger=True,sync_dist=True)
        self.log('num_fg',result['num_fg'],logger=True,sync_dist=True)
        
        return loss

    def training_step(self, batch, batch_nb):
        loss = self.step(batch)
        self.log('loss',loss,logger=True)
        return {'loss': loss}
    
    
    def validation_step(self, batch, batch_nb):
        
        
        img, box,cls,_,_  = batch
        
        outputs = self.base(img)
        outputs = postprocess(outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre)

        
        det_boxes = []
        gt_boxes_all = []
        for _ in range(7):
            det_boxes.append([])
        
        for b in range(len(outputs)):
            if outputs[b]==None:
                pass
            else:
                outputs_batch = outputs[b].cpu().numpy()
                box_batch = box[b].cpu().numpy()
                cls_batch = cls[b].cpu().numpy()[:,:7]
                
                
                frame_gt = get_individual_labels(box_batch, cls_batch)
                #print("Step_End:",outputs_batch.astype('int'))
                #print(frame_gt)
                gt_boxes_all.append(frame_gt)


                for cl_ind in range(7):
                    new_target = outputs_batch[outputs_batch[:,6]==cl_ind]
                    new_outputs = np.zeros((new_target.shape[0], 5))
                    new_outputs[:,0:4] = new_target[:,0:4]
                    new_outputs[:,4] =new_target[:,4]*new_target[:,5]
                    det_boxes[cl_ind].append(new_outputs)
                    #print(cl_ind,new_outputs.astype('int'))
        
        return det_boxes, gt_boxes_all
        
            
            
    def validation_epoch_end(self, validation_step_outputs):
        #print(len(validation_step_end_outputs))
        end_det_boxes=[]
        end_gt_boxes_all=[]
        for _ in range(7):
            end_det_boxes.append([])
            
        for det_boxes, gt_boxes_all in validation_step_outputs:
            for i in range(7):
                end_det_boxes[i]+=det_boxes[i]
            end_gt_boxes_all+= gt_boxes_all
            
            #print('Det_boxes:',len(det_boxes),len(det_boxes[0]))
            #print('====================')
            #print(len(det_boxes), len(gt_boxes_all))
        
        #print('Det_end_boxes:',len(end_det_boxes),len(end_det_boxes[0]))
          
        
        
        mAP, ap_all = evaluate.evaluate(end_gt_boxes_all, end_det_boxes, self.all_classes, iou_thresh=0.5)
        #print(mAP)
        #print('Epoch End:',len(end_gt_boxes_all),mAP[0])
        self.log('mAP',mAP[0],logger=True,sync_dist=True)
   

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(save_top_k=3,monitor='mAP',mode='max',filename='{epoch:02d}_{mAP:.3f}')
checkpoint_callback2 = ModelCheckpoint(save_last=True)



debug = False
gpus = torch.cuda.device_count()
tb_logger = pl.loggers.TensorBoardLogger('lightning_logs/')


#check_point = '/data/road-dataset/My_YOLOX/lightning_logs/default/version_95/checkpoints/last.ckpt'
#,reload_dataloaders_every_n_epochs=1
trainer = pl.Trainer(gpus=[0,1,2,3],accelerator='ddp',plugins=DDPPlugin(find_unused_parameters=False),num_sanity_val_steps=1,check_val_every_n_epoch=5,sync_batchnorm=True,logger=tb_logger,log_every_n_steps=2,precision=16, max_epochs=1500,callbacks=[lr_monitor, checkpoint_callback, checkpoint_callback2])
model = Model()
    
road = DataModuleRoad() 
trainer.fit(model, road)




#check_point = '/data/road-dataset/flat_Retina/lightning_logs/default/version_12/checkpoints/epoch=37_mAP=88.58005.ckpt'
#trainer = pl.Trainer(gpus=[0,1,2,3],distributed_backend='ddp', precision=16,resume_from_checkpoint=check_point,sync_batchnorm=True,logger=tb_logger,log_every_n_steps=2,num_sanity_val_steps=0,callbacks=[lr_monitor, checkpoint_callback, checkpoint_callback2])

