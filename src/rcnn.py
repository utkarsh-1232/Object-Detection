import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from torchvision.transforms.functional import resized_crop

class RCNNDataset(Dataset):
    def __init__(self, img_paths, rois, roi_ids, offsets, crop_size=(224,224)):
        self.img_paths, self.rois, self.roi_ids, self.offsets = img_paths, rois, roi_ids, offsets
        self.crop_size = crop_size
        self.img_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.img_tfms(img)
        _, H, W = img.shape
        offset = self.offsets[idx]/torch.tensor([W,H,W,H])
        x_min, y_min, w, h = self.rois[idx].int().tolist()
        crop = resized_crop(img, top=y_min, left=x_min, height=h, width=w, size=self.crop_size)
        return crop, self.roi_ids[idx], offset

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
        self.cls_loss_func = nn.CrossEntropyLoss()
        self.reg_head = nn.Sequential(
            nn.Linear(encode_dim, 512), nn.ReLU(),
            nn.Linear(512, 4), nn.Tanh(),
        )
        self.reg_loss_func = nn.MSELoss()

    def forward(self, crops):
        features = self.img_encoder(crops)
        probs = self.cls_head(features)
        bbox = self.reg_head(features)
        return probs, bbox

    def calc_loss(self, preds, ids, offsets, beta=0.2):
        probs, bbox = preds
        cls_loss = self.cls_loss_func(probs, ids)
        mask = ids!=0
        bbox, offsets = bbox[mask], offsets[mask]
        reg_loss = self.reg_loss_func(bbox, offsets) if len(mask)>0 else torch.tensor(0.0, requires_grad=True)
    
        print(cls_loss, reg_loss)
        return beta*cls_loss + (1-beta)*reg_loss