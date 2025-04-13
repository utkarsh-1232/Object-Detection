import pandas as pd
from src.core import perform_nms, calc_mAP

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2 as v2
from torchvision import models

from fastai.vision.all import DataLoaders, Metric

class RCNNDataset(Dataset):
    def __init__(self, roi_file, img_folder, id2img, crop_size=(224,224), augs=None):
        self.df = pd.read_json(roi_file)
        self.id2img = id2img
        self.tfms = v2.Compose([
            v2.Resize(size=crop_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if augs:
            self.tfms = v2.Compose(self.tfms.transforms[:-1]+augs+self.tfms.transforms[-1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        roi, roi_id, offset, img_id = self.df.iloc[idx]
        img = read_image(self.id2img[img_id], mode=ImageReadMode.RGB)
        img_id, roi_id = torch.tensor(img_id, dtype=torch.long), torch.tensor(roi_id, dtype=torch.long)
        roi, offset = torch.tensor(roi), torch.tensor(offset)
        x, y, w, h = roi.int()
        crop = img[:, y:y+h, x:x+w]
        return self.tfms(crop), img_id, roi, roi_id, offset

def get_dls(roi_folder, img_folder, id2img, crop_size=(224,224), train_sampler=None, augs=None, bs=64):
    train_roi_file = f"{roi_folder}/train_rois.json"
    train_ds = RCNNDataset(train_roi_file, img_folder, id2img, crop_size, augs)
    valid_roi_file = f"{roi_folder}/valid_rois.json"
    valid_ds = RCNNDataset(valid_roi_file, img_folder, id2img, crop_size)

    if train_sampler:
        train_dl = DataLoader(train_ds, batch_size=bs, sampler=train_sampler, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, pin_memory=True)
    
    dls = DataLoaders(train_dl, valid_dl)
    dls.n_inp = 1
    return dls

class RCNNModel(nn.Module):
    def __init__(self, n_classes):
        super(RCNNModel, self).__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg16 = vgg16.features
        self.adaptivepool = nn.Sequential(vgg16.avgpool, nn.Flatten())
        self.encoder = nn.Sequential(
            nn.Linear(25088, 4096, bias=False),
            nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, 512, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.cls_head = nn.Linear(512, n_classes+1)
        self.reg_head = nn.Linear(512, n_classes*4)

    def forward(self, crops):
        features = self.vgg16(crops)
        features = self.adaptivepool(features)
        encoded = self.encoder(features)
        probs = self.cls_head(encoded)
        pred_offsets = self.reg_head(encoded)
        return probs, pred_offsets

def reg_loss(preds, *targs):
    _, pred_offsets = preds
    _, _, roi_ids, offsets = targs
    if not torch.any(roi_ids!=0):
        return torch.tensor(0.0, device=roi_ids.device, requires_grad=True)
    roi_ids, offsets, pred_offsets = [
        t[roi_ids!=0] for t in [roi_ids, offsets, pred_offsets]
    ]
    bs = pred_offsets.size(0)
    pred_offsets = pred_offsets.view(bs,-1,4)[torch.arange(bs), roi_ids-1]
    return nn.SmoothL1Loss()(pred_offsets, offsets)

def cls_loss(preds, *targs):
    return nn.CrossEntropyLoss()(preds[0], targs[2])

def detn_loss(preds, *targs, beta=1):
    return cls_loss(preds, *targs) + beta*reg_loss(preds, *targs)

def apply_offsets(rois, pred_offsets):
    # Convert ROIs to center coordinates
    centers = rois[:, :2] + rois[:, 2:]/2
    widths = rois[:, 2]
    heights = rois[:, 3]
    
    # Unpack offsets
    dx = pred_offsets[:, 0]
    dy = pred_offsets[:, 1]
    dw = pred_offsets[:, 2]
    dh = pred_offsets[:, 3]
    
    # Apply transformations
    refined_centers_x = centers[:, 0] + dx * widths
    refined_centers_y = centers[:, 1] + dy * heights
    refined_widths = widths * torch.exp(dw)
    refined_heights = heights * torch.exp(dh)
    
    # Convert back to [x, y, w, h] format
    refined_boxes = torch.stack([
        refined_centers_x - refined_widths/2,
        refined_centers_y - refined_heights/2,
        refined_widths,
        refined_heights
    ], dim=1)
    
    return refined_boxes

def get_preds(img_ids, rois, probs, pred_offsets):
    probs = nn.Softmax(dim=1)(probs)
    scores, pred_ids = probs.max(dim=1)
    # filter background
    img_ids, rois, scores, pred_ids, pred_offsets = [
        t[pred_ids!=0] for t in [img_ids, rois, scores, pred_ids, pred_offsets]
    ]
    # get predicted bbs
    bs = pred_offsets.size(0)
    pred_offsets = pred_offsets.view(bs,-1,4)[torch.arange(bs), pred_ids-1]
    pred_bbs = apply_offsets(rois, pred_offsets)

    return img_ids, pred_ids, pred_bbs, scores

class mAPMetric(Metric):
    def __init__(self, gt_path, pred_path, nms_thresh=0.5):
        self.gt_path, self.pred_path = gt_path, pred_path
        self.nms_thresh, self.df_list = nms_thresh, []
        
    def reset(self):
        self.df_list = []

    def accumulate(self, learn):
        probs, pred_offsets = learn.pred
        img_ids, rois, _, _  = learn.y
        img_ids, pred_ids, pred_bbs, scores = \
        get_preds(img_ids, rois, probs, pred_offsets)
        batch_df = pd.DataFrame({
            'image_id': img_ids.cpu().tolist(),
            'category_id': pred_ids.cpu().tolist(),
            'bbox': pred_bbs.cpu().tolist(),
            'score': scores.cpu().tolist()
        })
        self.df_list.append(batch_df)
            
    @property
    def value(self):
        pred_df = pd.concat(self.df_list, ignore_index=True)
        pred_df = (
            pred_df
            .groupby('image_id').agg(list).reset_index()
            .apply(perform_nms, axis=1, thresh=self.nms_thresh)
            .explode(pred_df.columns.drop('image_id').tolist())
            .dropna(how='any')
        )
        pred_df.to_json(self.pred_path, orient='records')
        return calc_mAP(self.gt_path, self.pred_path)

def rcnn_splitter(model):
    body = list(model.vgg16.parameters()) + list(model.adaptivepool.parameters())
    heads = list(model.encoder.parameters()) + \
            list(model.cls_head.parameters()) + \
            list(model.reg_head.parameters())
    return body, heads