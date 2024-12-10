import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
from data import datasets
import albumentations as A
from data import custum_collate
import numpy as np
import glob, cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import albumentations.pytorch
import utility
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import os
import random
from yolox.exp import Exp
from yolox.utils import postprocess,crop_mask
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils
import datetime
from shutil import copyfile
import time
import modules.evaluation as evaluate

import torch.nn.functional as F
import scipy


# Initialize Distributed Environment
dist.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


base_seed = 42  # This can be any number
process_specific_seed = base_seed + local_rank

# Set the seed
torch.manual_seed(process_specific_seed)
np.random.seed(process_specific_seed)
random.seed(process_specific_seed)

#--------------Base Param------------
w_h = [1280, 480]
BATCH_SIZE = 4
num_workers =8
file_root = '/home/user/Dataset/ETRI_Dataset/'
#--------------Base Param------------
#--------------Class Param------------
all_classes =  ['Ped', 'Cyc', 'Mobike', 'Car', 'Bus', 'Cone', 'TL','VehLane', 'OutgoLane', 'MiddleLane', 'IncomLane',  'Pav', 'Jun', 'Xing_L', 'BusStop', 'Parking_L','Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Blocking', 'Informing', 'Brake', 'Stop', 'IncatLft',
                   'IncatRht', 'HazLit', 'HeadingLft', 'HeadingRht', 'Parking', 'EmVeh', 'School', 'Control', 'Xing']
len_class = len(all_classes)
agent_classes =  ['Ped', 'Cyc', 'Mobike', 'Car', 'Bus', 'Cone', 'TL']
loc_classes =  ['VehLane', 'OutgoLane', 'MiddleLane', 'IncomLane',  'Pav', 'Jun', 'Xing_L', 'BusStop', 'Parking_L']
action_classes =  ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Blocking', 'Informing', 'Brake', 'Stop', 'IncatLft',
                   'IncatRht', 'HazLit', 'HeadingLft', 'HeadingRht', 'Parking', 'EmVeh', 'School', 'Control', 'Xing']
class_nums = [len(agent_classes), len(loc_classes), len(action_classes)]
#--------------Class Param------------

#--------------Learning Rate Param------------
T_0=50# Initial Cycle Length
T_mult=1 # Cycle Length multiplier
eta_max=5e-4 # Max LR
T_up=2  # WarmUp Length
gamma=0.2 # LR reducer

start_epoch =0
num_epochs =151
best_acc = -1

val_cycle = 3
#--------------Learning Rate Param------------
#Train transfrom + Augmentations

train_transform =  A.Compose(
    [
     
     A.RandomSizedBBoxSafeCrop (width=w_h[0],height=w_h[1],erosion_rate=0.3,p=0.7),
         
     #A.Resize(width=w_h[0],height=w_h[1],p=1),
     A.OneOf([
         A.Sharpen(p=0.5),
         A.AdvancedBlur(p=0.5),
     ],p=0.66),
     A.RandomBrightnessContrast(p=0.5),

    A.Normalize(mean=(0,0,0),std=(1,1,1)),
    A.pytorch.transforms.ToTensorV2()],
    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
)


train_dataset = datasets.Infra_dataset(train=True,class_nums = class_nums,transform=train_transform,file_root = file_root)


#Val transfrom + Augmentations

val_transform =A.Compose(
    [
    A.Normalize(mean=(0,0,0),std=(1,1,1)),
    #A.Normalize(),
    A.pytorch.transforms.ToTensorV2()],
    bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']),
)
val_dataset = datasets.Infra_dataset(train=False,class_nums = class_nums,self_train=False,transform=val_transform,file_root = file_root)


# Your existing code for dataset preparation, model initialization, etc.

# Replace DataLoader with one using a DistributedSampler
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_data_loader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                          shuffle=False, num_workers=num_workers,
                                          collate_fn=custum_collate, pin_memory=True, 
                                          drop_last=True, sampler=train_sampler)

val_sampler = DistributedSampler(val_dataset)
val_data_loader = data_utils.DataLoader(val_dataset, batch_size=BATCH_SIZE//2, 
                                        shuffle=False, num_workers=num_workers//2,
                                        collate_fn=custum_collate, pin_memory=True, 
                                        drop_last=True, sampler=val_sampler)


exp = Exp(class_nums = class_nums)
model = exp.get_model()
head_feat = int(256 * exp.width)

#ckpt = torch.load('/home/etri/road-dataset/Track_YOLOX/last.ckpt')
#model = utility.load_my_state_dict(model,ckpt["state_dict"],legacy=True)

#ckpt = torch.load('./yolox_l.pth')
#model = utility.load_my_state_dict(model,ckpt["model"])

ckpt = torch.load('./output/Overfit/checkpoints/last.pth.tar')
model = utility.load_my_state_dict(model,ckpt["state_dict"])
#ckpt = torch.load('./output/2024-03-27-17-08-41/checkpoints/last.pth.tar')
#model = utility.load_my_state_dict(model,ckpt["state_dict"])
model.to(device)


# Wrap model with DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)


optimizer = torch.optim.AdamW(model.parameters(),lr=0)
scheduler = utility.CosineAnnealingWarmUpRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_max=eta_max,  T_up=T_up, gamma=gamma)
# Your existing code for the training loop

scaler = torch.cuda.amp.GradScaler()

# Ensure only the first process saves the model and logs
if local_rank == 0:
        #Save Path Creation
    now = datetime.datetime.now()
    output_dir = f'./output/{now.strftime("%Y-%m-%d-%H-%M-%S")}'
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    copyfile('./yolox/models/yolo_head.py', output_dir + '/'  + "yolo_head.py")
    
    # Save checkpoints, log information, etc.

for epoch in range(start_epoch, num_epochs):

    train_sampler.set_epoch(epoch)

    model.train()
    if local_rank == 0:
        print('training_start : ', epoch)
        start_time = time.time()
        
    for batch_idx, (img,mask, box,cls,_,_) in enumerate(train_data_loader):
        #try:
        img,mask, box,cls =  img.cuda(non_blocking=True),mask.cuda(non_blocking=True), box.cuda(non_blocking=True),cls.cuda(non_blocking=True)
        #print(box)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            result = model(img,mask, box,cls)
            loss = result['total_loss']
        scaler.scale(torch.sum(loss)).backward()
        scaler.step(optimizer)
        scaler.update()

        # Log training statistics to TensorBoard
        if local_rank == 0:
            n_iter = epoch * len(train_data_loader) + batch_idx
            writer.add_scalar('Train/Loss', torch.sum(result['total_loss']).item(), n_iter)
            writer.add_scalar('Train_Loss/Loss_reid', torch.sum(result['reid_loss']).item(), n_iter)
            writer.add_scalar('Train_Loss/Loss_iou',torch.sum(result['iou_loss']).item(), n_iter)
            writer.add_scalar('Train_Loss/Loss_conf', torch.sum(result['conf_loss']).item(), n_iter)
            writer.add_scalar('Train_Loss/Loss_cls', torch.sum(result['cls_loss']).item(), n_iter)
            writer.add_scalar('Train_Loss/Loss_contra', torch.sum(result['contra_loss']).item(), n_iter)

        #except:
        #    print(mimages.shape, fimages.shape,bimages.shape, boxes, labels)

    scheduler.step()    
    
    if local_rank == 0:
        writer.add_scalar('Train/Learning_rate', scheduler.get_lr()[0], epoch)
    
        print('training_end : ', epoch)
        end_time = time.time()
        print('Time elapse: ', end_time-start_time)

    if epoch%val_cycle>0:
        continue

    
    if local_rank == 0:
        print('Validation_start : ', epoch)
        start_time = time.time()
    
    model.eval()
    
    det_boxes = []
           
    gt_boxes_all = []
    num_instance = 0
    num_success_fifty = 0
    num_success_ninety = 0
    
    for _ in range(len_class):
        det_boxes.append([])
        

    with torch.no_grad():
        for batch_idx, (img,mask, box,cls,_,_) in enumerate(val_data_loader):
            #try:
            img,mask, box,cls =  img.cuda(non_blocking=True),mask.cuda(non_blocking=True), box.cuda(non_blocking=True),cls.cuda(non_blocking=True)
            #print(box.shape,cls.shape)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(1),size=(mask.shape[1]//4,mask.shape[2]//4),
                                                 mode='nearest-exact').squeeze(1)
            
            output = model(img)
            result,re_mask,bg_vector = output['results'],output['mask_output'],output['bg_vector']
            #print(result.shape)
            outputs = utility.val_postprocess(result, len_class, 0.1,0.5,head_feat)

            for b in range(len(outputs)):
                if b%2==0:
                    #print(len(det_boxes))
                    #print(len(gt_boxes_all))
                    if outputs[b]==None:
                        pass
                    else:
                        outputs_batch = outputs[b].cpu().numpy()
                        box_batch = box[b].cpu().numpy()
                        cls_batch = cls[b].cpu().numpy()[:,:len_class]


                        frame_gt = utility.get_individual_labels(box_batch, cls_batch)
                        #print("Step_End:",outputs_batch.astype('int'))
                        #print(frame_gt)
                        gt_boxes_all.append(frame_gt)

                        nframe_gt = utility.get_individual_labels(box_batch, cls_batch[:,:7])

                        pair_box = utility.pair_boxes(nframe_gt,outputs_batch[:,:6])


                        for cl_ind in range(len_class):
                            new_target = outputs_batch[outputs_batch[:,7+cl_ind]==1]
                            new_outputs = np.zeros((new_target.shape[0], 5))
                            new_outputs[:,0:4] = new_target[:,0:4]
                            new_outputs[:,4] =new_target[:,4]*new_target[:,5]
                            det_boxes[cl_ind].append(new_outputs)


                        iouEvalVal = utility.iouEval(nframe_gt.shape[0]+1)

                        target_mask = re_mask[b,:,:,:]      
                        target_vec = F.normalize(torch.cat((outputs[b][:,-head_feat:],bg_vector[b].unsqueeze(0)),dim=0),p=2,dim=1)

                        semseg = torch.einsum("cq,qhw->chw", target_vec, target_mask)

                        max_val, max_idx= torch.max(semseg,0)

                        target_mask = torch.zeros_like(max_idx)
                        for gt, target in pair_box:
                            target_mask[max_idx==target]=gt+1

                        iouEvalVal.addBatch(target_mask.unsqueeze(0).unsqueeze(0).to(torch.long), mask[b].unsqueeze(0).unsqueeze(0).to(torch.long))

                        iouTrain, iou_classes = iouEvalVal.getIoU()
                        #print(iouTrain, iou_classes)
                        num_instance += len(iou_classes)
                        num_success_fifty += len(iou_classes[iou_classes>0.5])
                        num_success_ninety += len(iou_classes[iou_classes>0.9])
                        #print(num_instance, num_success)
            
    

        #mAP, ap_all = evaluate.evaluate(gt_boxes_all,det_boxes, all_classes, iou_thresh=0.5)

        mAP, ap_all = evaluate.evaluate(gt_boxes_all,det_boxes, all_classes, iou_thresh=0.5)
        mAP = mAP[0]
        accuracy50 = num_success_fifty/(num_instance+0.1)
        accuracy90 = num_success_ninety/(num_instance+0.1)
        mAP = torch.tensor([mAP], device=device)
        accuracy50 = torch.tensor([accuracy50], device=device)
        accuracy90 = torch.tensor([accuracy90], device=device)

        dist.all_reduce(mAP, op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy50, op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy90, op=dist.ReduceOp.SUM)

        
        if local_rank == 0:
            
            combined_mAP =  mAP / dist.get_world_size()
            combined_accuracy50 =  accuracy50 / dist.get_world_size()
            combined_accuracy90 =  accuracy90 / dist.get_world_size()
            
            print(combined_mAP, combined_accuracy50, combined_accuracy90)
            writer.add_scalar('Val/Accuracy50', combined_accuracy50, epoch)
            writer.add_scalar('Val/Accuracy90', combined_accuracy90, epoch)
            writer.add_scalar('Val/mAP', combined_mAP, epoch)

            accuracy =(combined_accuracy50*100+combined_accuracy90*100+combined_mAP)/3
            print(accuracy)
            is_best = best_acc<accuracy
            if is_best:
                best_acc = accuracy
                print('Best: ', accuracy)
            last_PATH = checkpoint_dir+'/last.pth.tar'
            best_PATH = checkpoint_dir+'/best.pth.tar'
            utility.save_checkpoint({
                'epoch': epoch+1,
                'arch': str(model),
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                #'optimizer' : optimizer.state_dict(),
            }, is_best, last_PATH, best_PATH)


    if local_rank == 0:
        print('Val_end : ', epoch)
        end_time = time.time()
        print('Time elapse: ', end_time-start_time)
        
# Finalize the distributed environment at the end of your script
dist.destroy_process_group()
