import io, sys, json
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
from src.core import load_data
from src.rois import apply_offsets, get_annotated_rois

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2 as v2
from torchvision import models, ops

from fastai.vision.all import DataLoaders, Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class RCNNDataset(Dataset):
    def __init__(self, ann_file, imgs_folder, df_sample=1, class_sampler=None,
                 crop_size=(224,224), tfms=None, **ann_kwargs):
        self.crop_size = crop_size
        self.tfms = tfms
        if self.tfms is None:
            self.tfms = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        df, self.id2label, self.id2img = load_data(ann_file, imgs_folder)
        df = df.sample(frac=df_sample)
        res = df.progress_apply(get_annotated_rois, axis=1, id2img=self.id2img, **ann_kwargs)
        self.img_ids = torch.cat([row[0].repeat(row[1].shape[0]) for row in res])
        self.rois = torch.cat([row[1] for row in res])
        self.roi_ids = torch.cat([row[2] for row in res])
        self.offsets = torch.cat([row[3] for row in res])

        self.idxs = torch.tensor(range(len(self.roi_ids)))[:, None]
        if class_sampler: self.idxs = class_sampler(self.roi_ids)

    def __len__(self): return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx].item()
        img_id = self.img_ids[idx]
        img = read_image(self.id2img[img_id.item()], mode=ImageReadMode.RGB)
        img = self.tfms(img)
        x_min, y_min, w, h = self.rois[idx].int().tolist()
        crop = v2.functional.resized_crop(img, y_min, x_min, h, w, self.crop_size)
        
        return crop, img_id, self.rois[idx], self.roi_ids[idx], self.offsets[idx]

def get_dls(train_ds, valid_ds, bs=64):
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
        loss = nn.L1Loss()(pred_offsets[mask], offsets[mask])
    return loss

def cls_loss(preds, *targs):
    return nn.CrossEntropyLoss()(preds[0], targs[2])

def detn_loss(preds, *targs):
    return cls_loss(preds, *targs) + reg_loss(preds, *targs)

def perform_nms(row, thresh=0.5):
    ids, bbs, scores = row[['category_id','bbox','score']]
    ids, bbs, scores = [torch.tensor(l) for l in [ids, bbs, scores]]
    ids, bbs, scores = [t[ids!=0] for t in [ids, bbs, scores]]
    boxes = torch.cat([bbs[:,:2], bbs[:,:2]+bbs[:,2:]-1], dim=1)
    idxs = ops.batched_nms(boxes, scores, ids, thresh)
    row[['category_id','bbox','score']] = [t[idxs].tolist() for t in [ids, bbs, scores]]
    return row

def calc_map(gt_path, pred_path):
    with io.StringIO() as buf:
        save_stdout = sys.stdout
        sys.stdout = buf  # Redirect standard output
        coco_gt = COCO(gt_path)
        coco_pred = coco_gt.loadRes(pred_path)
        cocoEval = COCOeval(coco_gt, coco_pred, 'bbox')
        cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
        sys.stdout = save_stdout  # Restore standard output
    return cocoEval.stats[0]

class mAP(Metric):
    def __init__(self, gt_path, pred_path, nms_thresh=0.5):
        self.gt_path, self.pred_path, self.nms_thresh = gt_path, pred_path, nms_thresh
        self.reset()
        
    def reset(self): self.df_list = []

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
        exp_df.to_json(self.pred_path, orient='records')
        return calc_map(self.gt_path, self.pred_path)