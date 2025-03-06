from src.rois import ROIs
import torch
import torch.nn as nn
from torchvision import models
from selectivesearch import selective_search

def extract_rois_ss(img):
    img_area = img.shape[1]*img.shape[2]
    _, regions = selective_search(img.permute((1,2,0)), scale=200, min_size=100)
    rois = torch.tensor([r['rect'] for r in regions])
    sizes = torch.tensor([r['size'] for r in regions])
    mask = (sizes>0.05*img_area) & (sizes<img_area)
    return ROIs(rois[mask, :])

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
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_head = nn.Sequential(
             nn.Linear(encode_dim, 512), nn.ReLU(),
             nn.Linear(512, 4), nn.Sigmoid(),
        )
        self.reg_loss = nn.L1Loss()

    def forward(self, crops):
        features = self.img_encoder(crops)
        probs = self.cls_head(features)
        bbox = self.reg_head(features)
        return probs, bbox

    def calc_loss(self, preds, ids, offsets, beta=0.1):
        probs, bbox = preds
        cls_loss = self.cls_loss(probs, ids)
        mask = ids!=0
        bbox, offsets = bbox[mask], offsets[mask]
        reg_loss = self.reg_loss(bbox, offsets) if len(mask)>0 else torch.tensor(0.0, requires_grad=True)

        return beta*cls_loss + (1-beta)*reg_loss