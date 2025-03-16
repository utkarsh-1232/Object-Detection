import io, sys, json
from pathlib import Path
from functools import partial
from tqdm import tqdm
tqdm.pandas()
from src.core import load_data
from src.rois import apply_offsets, get_annotated_rois

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import one_hot
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2 as v2
from torchvision import models
from torchvision.transforms.v2.functional import resized_crop

from fastai.vision.all import DataLoaders, Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class RCNNDataset(Dataset):
    def __init__(self, ann_path, imgs_path, sample_frac=1, crop_size=(224,224), tfms=None):
        ann_path = Path(ann_path)
        df, self.id2label, self.id2img = load_data(str(ann_path.parent), imgs_path, ann_path.stem)
        replace = True if sample_frac>1 else False
        df = df.sample(frac=sample_frac, replace=replace)
        self.crop_size = crop_size
        self.tfms = tfms
        if self.tfms is None:
            self.tfms = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        res = df.progress_apply(get_annotated_rois, axis=1, id2img=self.id2img)
        self.img_ids = torch.cat([row[0] for row in res])
        self.rois = torch.cat([row[1] for row in res])
        self.roi_ids = torch.cat([row[2] for row in res])
        self.offsets = torch.cat([row[3] for row in res])

    def __len__(self):
        return self.img_ids.shape[0]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = io.read_image(self.id2img[img_id.item()], mode=io.ImageReadMode.RGB)
        img = self.tfms(img)
        x_min, y_min, w, h = self.rois[idx].int().tolist()
        crop = resized_crop(img, top=y_min, left=x_min, height=h, width=w, size=self.crop_size)
        
        return crop, img_id, self.rois[idx], self.roi_ids[idx], self.offsets[idx]

def get_dls(train_ds, valid_ds, bs=128, tfms=None):
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, pin_memory=True)
    
    dls = DataLoaders(train_dl, valid_dl)
    dls.n_inp = 1
    return dls

class RCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        encode_dim = self.model.classifier[0].in_features

        head = nn.Sequential(
            nn.Linear(encode_dim, 4096), nn.ReLU(),
            nn.BatchNorm1d(4096), nn.Dropout(0.5),
            nn.Linear(4096, 512), nn.ReLU(),
            nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, n_classes+5)
        )
        self.model.classifier = head

    def forward(self, crops): return self.model(crops)

def reg_loss(preds, *targs):
    _, _, roi_ids, offsets = targs
    loss = torch.tensor(0.0, requires_grad=True)
    mask = roi_ids!=0
    if torch.sum(mask)>0:
        loss = nn.L1Loss()(preds[mask, -4:], offsets[mask])
    return loss

def cls_loss(preds, *targs):
    _, _, roi_ids, _ = targs
    loss = torch.tensor(0.0, requires_grad=True)
    n_classes = preds.shape[1]-4
    cats = one_hot(roi_ids, num_classes=n_classes)
    preds[:,:-4] = nn.Sigmoid()(preds[:,:-4])
    loss = nn.BCELoss()(preds[:,1:-4], cats[:,1:].to(torch.float32))
    return loss

def detn_loss(preds, *targs):
    return cls_loss(preds, *targs) + reg_loss(preds, *targs)

class mAP(Metric):
    def __init__(self, gt_path, pred_path):
        self.gt_path, self.pred_path = gt_path, pred_path
        self.reset()
        
    def reset(self):
        with open(self.pred_path, "w") as f:
            json.dump([], f, indent=4)

    def accumulate(self, learn):
        probs, pred_offsets = learn.pred[:,:-4], learn.pred[:,-4:]
        scores, pred_ids = probs.max(dim=1)
        img_ids, rois, roi_ids, _  = learn.y
        mask = roi_ids!=0
        pred_bbs = apply_offsets(rois, pred_offsets)
        
        self.write_to_file(img_ids[mask], pred_ids[mask], pred_bbs[mask], scores[mask])

    def write_to_file(self, img_ids, ids, pred_bbs, scores):
        iterables = [img_ids.tolist(), ids.tolist(), pred_bbs.tolist(), scores.tolist()]
        new_anns = [
            {'image_id':img_id, 'category_id':id, 'bbox':bbox, 'score':score}
            for img_id, id, bbox, score in zip(*iterables)
        ]
        with open(self.pred_path, 'r') as f:
            data = json.load(f)
        data += new_anns
        with open(self.pred_path, 'w') as f:
            json.dump(data, f, indent=4)
            
    @property
    def value(self):
        with io.StringIO() as buf:
            save_stdout = sys.stdout
            sys.stdout = buf  # Redirect standard output
            coco_gt = COCO(self.gt_path)
            coco_pred = coco_gt.loadRes(self.pred_path)
            cocoEval = COCOeval(coco_gt, coco_pred, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            sys.stdout = save_stdout  # Restore standard output
        return cocoEval.stats[0]