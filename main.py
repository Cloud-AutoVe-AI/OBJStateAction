from data import datasets
import albumentations as A
from data import custum_collate
import numpy as np
import glob, cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import albumentations.pytorch

import torch
import torch.distributed as dist
import torch.nn as nn
import sys
import torchvision
import os
import random
import scipy
import time

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
#from yolox.utils import postprocess


MEANS =[0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]
w_h = [1280,480]
transform = A.Compose(
    [
     A.Normalize(mean=MEANS,std=STDS),
    A.pytorch.transforms.ToTensorV2()]
)
save_transform = A.Compose(
    [
        # A.Crop(x_min=38,y_min=25,x_max=1242,y_max=780),
        # A.Resize(width=w_h[0],height=w_h[1]),
        # A.ShiftScaleRotate(p=0.5),

        A.pytorch.transforms.ToTensorV2()],

)

agent_list = ['Ped', 'Cyc', 'Mobike', 'Car', 'Bus', 'Cone', 'TL']
loc_list = ['VehLane', 'OutgoLane', 'MiddleLane', 'IncomLane',  'Pav', 'Jun', 'Xing_L', 'BusStop', 'Parking_L']
action_list = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Blocking', 'Informing', 'Brake', 'Stop', 'IncatLft',
               'IncatRht', 'HazLit', 'HeadingLft', 'HeadingRht', 'Parking', 'EmVeh', 'School', 'Control', 'Xing']

icons = {}
for actions in action_list:
    target = './Icons/' + actions + '.png'
    icon_img = cv2.imread(target)
    icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
    icons[actions] = icon_img

for actions in loc_list:
    # print(actions)
    target = './Icons/' + actions + '.png'
    icon_img = cv2.imread(target)
    icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
    icons[actions] = icon_img


def postprocess(prediction, num_classes, conf_thre=0.1, nms_thre=0.85):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5: 5 + 7], 1, keepdim=True
        )
        loc_conf, loc_pred = torch.max(
            image_pred[:, 5 + 7: 5 + 7 + 9], 1, keepdim=True
        )
        action_conf, action_pred = torch.max(
            image_pred[:, 5 + 7 + 9: 5 + 7 + 9 + 19], 1, keepdim=True
        )
        reid_feat = image_pred[:, -128:]

        # print(class_conf[:,0].shape)
        # print('----------------------------')
        # print('class',np.min(class_conf[:,0].cpu().numpy()),np.max(class_conf[:,0].cpu().numpy()),np.mean(class_conf[:,0].cpu().numpy()))
        # print('loc',np.min(loc_conf[:,0].cpu().numpy()),np.max(loc_conf[:,0].cpu().numpy()),np.mean(loc_conf[:,0].cpu().numpy()))
        # print('action',np.min(action_conf[:,0].cpu().numpy()),np.max(action_conf[:,0].cpu().numpy()),np.mean(action_conf[:,0].cpu().numpy()))
        action_idx = (image_pred[:, 5 + 7 + 9: 5 + 7 + 9 + 19] + 0.6) // 1

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() * loc_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf * loc_conf, class_pred.float(), loc_pred.float(), action_idx, reid_feat), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            torch.zeros(detections.shape[0]),
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


#def get_model(depth=0.67, width=0.75, num_classes=35):
#def get_model(depth=0.33, width=0.5, num_classes=35):
def get_model(depth=1, width=1, num_classes=35):
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model

model = get_model()

model = model.cuda()


def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()

    for name, param in state_dict.items():
        name = name[5:]
        if name not in own_state:
            print('not loaded:', name)
            continue
        else:
            try:

                param.requires_grad = True
                own_state[name].copy_(param)
                # print(own_state[name].requires_grad)
                # print('Loaded : ', name)
            except:
                # param.requires_grad = True
                print('Imcompatible: ', name)

    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        # print(name, param.requires_grad)
    return model

#ckpt = torch.load('/data/road-dataset/My_YOLOX/yolox_s.pth.tar')
#model = load_my_state_dict(model,ckpt["model"])
ckpt = torch.load('./model/last.ckpt')
model = load_my_state_dict(model,ckpt["state_dict"])
model.eval()


def plot_one_box(x, img, cls, loc, action, index, color=None, line_thickness=None, icons=None):
    # Plots one bounding box on image img
    if int(index) == 0:
        pass
    else:
        tl = line_thickness or 2  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        offset = 2

        agent_list = ['Ped', 'Cyc', 'Mobike', 'Car', 'Bus', 'Cone', 'TL']
        loc_list = ['VehLane', 'OutgoLane', 'MiddleLane', 'IncomLane', 'Pav', 'Jun', 'Xing_L', 'BusStop', 'Parking_L']
        action_list = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Blocking', 'Informing', 'Brake', 'Stop',
                       'IncatLft',
                       'IncatRht', 'HazLit', 'HeadingLft', 'HeadingRht', 'Parking', 'EmVeh', 'School', 'Control',
                       'Xing']
        max_t = 0
        tf = 1  # font thickness
        t_size = cv2.getTextSize(loc_list[loc], 0, fontScale=tl / 3, thickness=tf)[0]
        max_t = max(max_t, t_size[0])

        if (index == 0):
            print("what...")
        cv2.putText(img, index, (c1[0], c2[1] + t_size[1] - offset), 0, tl / 3, COLORS[int(index) % 7], thickness=tf,
                    lineType=cv2.LINE_AA)

        num_icon = np.sum(action)
        if cls == 6:
            icon_size = int(np.min([(c2[0] - c1[0]) / (num_icon + 1), (x[3] - x[1]), 64]))
        elif cls == 0 or cls == 5:
            icon_size = int(np.min([(c2[0] - c1[0]) / num_icon, (x[3] - x[1]), 64]))

        else:
            icon_size = int(np.min([(c2[0] - c1[0]) / num_icon, (x[3] - x[1]) / 2, 64]))
        c3 = c1[0]  # +(c2[0]-c1[0])//2-icon_size*num_icon//2

        try:
            offset_icon = 0
            for ii in range(len(action)):
                if action[ii] == 1:
                    if cls == 0 or cls == 6 or cls == 5:
                        img[c1[1] - icon_size:c1[1], c3 + offset_icon:c3 + offset_icon + icon_size, :] = cv2.resize(
                            icons[action_list[ii]], (icon_size, icon_size),
                            interpolation=cv2.INTER_NEAREST) * 0.5 + img[c1[1] - icon_size:c1[1],
                                                                     c3 + offset_icon:c3 + offset_icon + icon_size,
                                                                     :] * 0.5
                    else:
                        img[c1[1]:c1[1] + icon_size, c3 + offset_icon:c3 + offset_icon + icon_size, :] = cv2.resize(
                            icons[action_list[ii]], (icon_size, icon_size),
                            interpolation=cv2.INTER_NEAREST) * 0.5 + img[c1[1]:c1[1] + icon_size,
                                                                     c3 + offset_icon:c3 + offset_icon + icon_size,
                                                                     :] * 0.5
                    offset_icon += icon_size

            if cls == 6:
                img[c1[1] - icon_size:c1[1], c3 + offset_icon:c3 + offset_icon + icon_size, :] = cv2.resize(
                    icons[loc_list[loc]], (icon_size, icon_size)) * 0.5 + img[c1[1] - icon_size:c1[1],
                                                                          c3 + offset_icon:c3 + offset_icon + icon_size,
                                                                          :] * 0.5
            else:
                img[c2[1] - icon_size:c2[1], c3:c3 + icon_size, :] = cv2.resize(icons[loc_list[loc]],
                                                                                (icon_size, icon_size)) * 0.5 + img[c2[
                                                                                                                        1] - icon_size:
                                                                                                                    c2[
                                                                                                                        1],
                                                                                                                c3:c3 + icon_size,
                                                                                                                :] * 0.5

        except:
            pass

        cv2.rectangle(img, c1, c2, color, thickness=1)


COLORS = [[255, 0, 0], [255, 215, 0], [199, 21, 133], [0, 255, 0], [0, 255, 255], [0, 0, 255], [30, 144, 255]]
# for _ in range(len(agent_list)):
#    COLORS.append([np.random.randint(0, 255) for _ in range(3)])


def linear_assignment(cost_matrix):
    x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


class BoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, cls_n, reid_feat):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.bbox = bbox[:4]
        self.reid_feat = reid_feat
        self.cls = cls_n
        self.time_since_update = 0
        self.id = BoxTracker.count
        BoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox, cls_n, feats):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox[:4]
        self.reid_feat = self.reid_feat * (self.hits / (self.hits + 1)) + feats * (1 / (self.hits + 1))
        self.cls = cls_n

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        return [self.bbox, self.cls, self.reid_feat]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return [np.concatenate((self.bbox, self.cls), 0)]


def associate_detections_to_trackers(detections, reid_feats, trackers, trks_feat, iou_threshold=0.3, blend_factor=0.5):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    reid_matrix = np.matmul(reid_feats, np.transpose(trks_feat))
    iou_matrix = iou_matrix * blend_factor + reid_matrix * (1 - blend_factor)

    # print("iou, reid: \n", iou_matrix,'\n', reid_matrix)
    # print(reid_feats.shape, trks_feat.shape)
    # print(iou_matrix)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class QSORT(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.1, blend_factor=0.5):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.blend_factor = blend_factor

    def update(self, dets=np.empty((0, 5)), cls=np.empty((0, 1 + 1 + 19)), reid_feats=np.empty((0, 128))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        trks_cls = np.zeros((len(self.trackers), 21))
        trks_feat = np.zeros((len(self.trackers), 128))

        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos, trk_cls, trk_feat = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            trks_cls[t, :] = trk_cls
            trks_feat[t, :] = trk_feat
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trks_cls = np.ma.compress_rows(np.ma.masked_invalid(trks_cls))
        trks_feat = np.ma.compress_rows(np.ma.masked_invalid(trks_feat))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, reid_feats, trks, trks_feat,
                                                                                   self.iou_threshold,
                                                                                   self.blend_factor)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], cls[m[0], :], reid_feats[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = BoxTracker(dets[i, :], cls[i, :], reid_feats[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5 + 21))


target_folder = './Dataset/00_Street/'
searchLabel = sorted(os.listdir(target_folder))

count = 0
frame_num = 0
# =========MOT20=============================
max_age = 5
min_hits = 3
iou_threshold = 0.05
blend_factor = 0.5
conf_thre = 0.5
nms_thre = 0.7
# =========MOT15=============================

# ======================================
reid_len = 128
# ======================================
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
write = False
print("Total: ", len(searchLabel))

mot_tracker = QSORT(max_age, min_hits, iou_threshold, blend_factor)
BoxTracker.count = 0
with torch.no_grad():
    for jj in range(len(searchLabel) - 1):
        if jj % 1 == 0:

            # ===============================
            t1.append(time.time())
            # ===============================

            img_name = target_folder + '/' + searchLabel[jj]
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            clip = transform(image=img)

            images = clip['image'].unsqueeze(0)

            height, width = images.shape[-2:]
            wh = [height, width]
            images = images.cuda(0, non_blocking=True)

            # ===============================
            t2.append(time.time())
            # ===============================
            outputs = model(images)
            # print(outputs[0].shape)
            # ===============================
            t3.append(time.time())
            # ===============================
            outputs = postprocess(outputs, 35, 0.01, 0.65)
            # print(outputs[0].shape)
            # ===============================
            t4.append(time.time())
            # ====================================================

            if outputs[0] == None:
                trackers = mot_tracker.update()
            else:
                outputs = outputs[0].cpu().numpy()
                # print("outputs shape:",outputs.shape)
                reid_feat = outputs[:, -reid_len:]
                cls_n = outputs[:, 6:-reid_len]
                dets = outputs[:, 0:5]
                # print('dets shape',dets.shape)
                trackers = mot_tracker.update(dets, cls_n, reid_feat)
            # print(trackers)

            #clip = save_transform(image=img)
            #s_img = clip['image']
            # print(s_img.shape)
            #img = s_img.numpy().transpose(1, 2, 0).astype(np.uint8).copy()
            # img = cv2.resize(img,(wh[1],wh[0]))
            # print(trackers.shape)
            if outputs == []:
                pass
            else:
                xyxy = trackers[:, 0:4]
                cls = trackers[:, 4].astype('int')
                loc = trackers[:, 5].astype('int')
                action = trackers[:, 6:-1].astype('int')
                ids = trackers[:, -1]
                # print(xyxy.shape, cls.shape, loc.shape, action.shape, ids.shape, trackers.shape)
                # print(xyxy.shape)
                for i in range(xyxy.shape[0]):
                    # print(i)

                    plot_one_box(xyxy[i], img, cls[i], loc[i], action[i], color=COLORS[cls[i]], index=str(int(ids[i])),
                                 icons=icons)

            # ===============================

            t5.append(time.time())
            # ===============================
            if write == True:
                img = img[:, :, ::-1].copy()

                path = './Result/' + str(jj).zfill(6) + '.png'
                cv2.imwrite(path, img)
            else:
                #print(t5[-1]-t1[-1])
                img = img[:, :, ::-1].copy()

                cv2.putText(img, "FPS" + "{: .2f}".format(1 / (t5[-1] - t1[-1])), (10, 29), 0, 1, [255, 255, 0],
                            thickness=2,
                            lineType=cv2.LINE_AA)

                cv2.imshow('image', img)
                key_in = cv2.waitKey(1)

            #print("\rProgress: {:>3} %".format(jj * 100 / len(searchLabel)), end=' ')
            sys.stdout.flush()