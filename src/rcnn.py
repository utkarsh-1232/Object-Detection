from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
from src.core import load_data, perform_nms, calc_mAP
from src.rois import apply_offsets, get_annotated_rois

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision import models

from fastai.vision.all import DataLoaders, Metric

class RCNNDataset(Dataset):
    def __init__(self, ann_file, imgs_folder, tfms, imgs_frac=1, **ann_kwargs):
        df, self.id2label, self.id2img = load_data(ann_file, imgs_folder)
        df = df.sample(frac=imgs_frac)
        
        res = df.progress_apply(get_annotated_rois, axis=1, id2img=self.id2img, **ann_kwargs)
        self.img_ids = torch.cat([row[0].repeat(row[1].shape[0]) for row in res])
        self.rois = torch.cat([row[1] for row in res])
        self.roi_ids = torch.cat([row[2] for row in res])
        self.offsets = torch.cat([row[3] for row in res])

        self.tfms = tfms
        self.idxs = torch.arange(len(self.roi_ids))[:, None]
            
    def sample_rois(self, sampler):
        self.idxs = sampler(self.roi_ids)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx].item()
        img_id = self.img_ids[idx]
        img = read_image(self.id2img[img_id.item()], mode=ImageReadMode.RGB)
        x, y, w, h = self.rois[idx].int().tolist()
        crop = img[:, y:y+h, x:x+w]
        
        return self.tfms(crop), img_id, self.rois[idx], self.roi_ids[idx], self.offsets[idx]

def get_dls(train_ds, valid_ds, bs=64, train_sampler=None):
    if train_sampler:
        train_ds.sample_rois(train_sampler)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, pin_memory=True)
    
    dls = DataLoaders(train_dl, valid_dl)
    dls.n_inp = 1
    return dls

class RCNNModel(nn.Module):
    def __init__(self, n_classes, p=0.5):
        super(RCNNModel, self).__init__()
        self.encoder = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        encode_dim = self.encoder.classifier[0].in_features
        self.encoder.classifier = nn.Sequential()
        
        self.cls_head = nn.Sequential(
            nn.Linear(encode_dim, 4096), nn.ReLU(),
            nn.Dropout(p), nn.Linear(4096, n_classes+1)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(encode_dim, 4096), nn.ReLU(),
            nn.BatchNorm1d(4096), nn.Dropout(p),
            nn.Linear(4096, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(p),
            nn.Linear(512, 4)
        )

    def forward(self, crops):
        features = self.encoder(crops)
        probs = self.cls_head(features)
        pred_offsets = self.reg_head(features)
        return probs, pred_offsets

def reg_loss(preds, *targs):
    _, pred_offsets = preds
    _, _, roi_ids, offsets = targs
    loss = torch.tensor(0.0, requires_grad=True)
    mask = roi_ids!=0
    if torch.sum(mask)>0:
        loss = nn.SmoothL1Loss()(pred_offsets[mask], offsets[mask])
    return loss

def cls_loss(preds, *targs):
    return nn.CrossEntropyLoss()(preds[0], targs[2])

def detn_loss(preds, *targs, beta=1):
    return cls_loss(preds, *targs) + beta*reg_loss(preds, *targs)

class mAPMetric(Metric):
    def __init__(self, gt_path, pred_path, nms_thresh=0.5):
        self.gt_path, self.pred_path, self.nms_thresh = gt_path, pred_path, nms_thresh
        self.reset()
        
    def reset(self):
        self.df_list = []

    def accumulate(self, learn):
        probs, pred_offsets = learn.pred
        probs = nn.Softmax(dim=1)(probs)
        scores, pred_ids = probs.max(dim=1)
        img_ids, rois, _, _  = learn.y
        pred_bbs = apply_offsets(rois.cpu(), pred_offsets.cpu()).numpy()
        batch_df = pd.DataFrame({
            'image_id': img_ids.cpu().numpy().tolist(),
            'category_id': pred_ids.cpu().numpy().tolist(),
            'bbox': pred_bbs.tolist(),
            'score': scores.cpu().numpy().tolist()
        })
        self.df_list.append(batch_df)
            
    @property
    def value(self):
        pred_df = pd.concat(self.df_list, ignore_index=True)
        agg_df = pred_df.groupby('image_id').agg(list).reset_index()
        nms_df = agg_df.apply(perform_nms, axis=1, thresh=self.nms_thresh)
        exp_df = nms_df.explode(nms_df.columns.drop('image_id').tolist())
        exp_df.dropna(how='any', inplace=True)
        exp_df.to_json(self.pred_path, orient='records')
        return calc_mAP(self.gt_path, self.pred_path)