from PIL import Image
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from torchvision.transforms.functional import resized_crop
from fastai.vision.all import Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class RCNNDataset(Dataset):
    def __init__(self, img_ids, rois, roi_ids, offsets, img_dims, id2img, crop_size=(224,224)):
        self.img_ids, self.rois, self.roi_ids = img_ids, rois, roi_ids
        self.offsets, self.img_dims = offsets, img_dims
        self.id2img = id2img
        self.crop_size = crop_size
        self.img_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.img_ids.shape[0]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = Image.open(self.id2img[img_id.item()]).convert('RGB')
        img = self.img_tfms(img)
        offset = self.offsets[idx]/self.img_dims[idx]
        x_min, y_min, w, h = self.rois[idx].int().tolist()
        crop = resized_crop(img, top=y_min, left=x_min, height=h, width=w, size=self.crop_size)
        
        return crop, img_id, self.rois[idx], self.roi_ids[idx], offset, self.img_dims[idx]

class RCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.img_encoder = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        self.img_encoder.eval()
        encode_dim = self.img_encoder.classifier[0].in_features
        self.img_encoder.classifier = nn.Sequential()

        self.cls_head = nn.Linear(encode_dim, n_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(encode_dim, 512), nn.ReLU(),
            nn.Linear(512, 4), nn.Tanh(),
        )

    def forward(self, crops):
        features = self.img_encoder(crops)
        probs = self.cls_head(features)
        bbox = self.reg_head(features)
        return probs, bbox

    def calc_loss(self, preds, *targs, beta=0.2):
        probs, pred_offsets = preds
        _, _, ids, offsets, _ = targs
        cls_loss = nn.CrossEntropyLoss()(probs, ids)
        reg_loss = torch.tensor(0.0, requires_grad=True)
        mask = ids!=0
        if torch.sum(mask)>0:
            reg_loss = nn.MSELoss()(pred_offsets[mask], offsets[mask])
            
        return beta*cls_loss + (1-beta)*reg_loss

class mAP(Metric):
    def __init__(self, gt_path, pred_path):
        self.gt_path, self.pred_path = gt_path, pred_path
        self.reset()
        
    def reset(self):
        with open(self.pred_path, "w") as f:
            json.dump([], f, indent=4)

    def accumulate(self, learn):
        probs, pred_offsets = learn.pred
        scores, pred_ids = probs.max(dim=1)
        img_ids, rois, ids, _, img_dims  = learn.y
        mask = ids!=0
        pred_offsets = pred_offsets*img_dims
        pred_bbs = rois+pred_offsets
        
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
        coco_gt = COCO(self.gt_path)
        coco_pred = coco_gt.loadRes(self.pred_path)
        cocoEval = COCOeval(coco_gt, coco_pred, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats[0]