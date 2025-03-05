from typing import Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop, resize

@dataclass
class ROIs:
    boxes: torch.FloatTensor
    ids: Optional[torch.IntTensor] = None
    offsets: Optional[torch.FloatTensor] = None
    ious: Optional[torch.FloatTensor] = None
    
    def __post_init__(self):
        self.ids = torch.full((self.boxes.shape[0],), 0, dtype=torch.long)

def get_ious(rois, bboxes, epsilon=1e-5):
    rois = torch.cat([rois[:,:2], rois[:,:2]+rois[:,2:]], dim=1)
    bboxes = torch.cat([bboxes[:,:2], bboxes[:,:2]+bboxes[:,2:]], dim=1)

    inter_xmin = torch.max(rois[:, None, 0], bboxes[None, :, 0])
    inter_ymin = torch.max(rois[:, None, 1], bboxes[None, :, 1])
    inter_xmax = torch.min(rois[:, None, 2], bboxes[None, :, 2])
    inter_ymax = torch.min(rois[:, None, 3], bboxes[None, :, 3])

    inter_w = (inter_xmax-inter_xmin).clamp(min=0)  # Ensure non-negative width
    inter_h = (inter_ymax-inter_ymin).clamp(min=0)  # Ensure non-negative height
    intersection = inter_w*inter_h
    
    area1 = (rois[:, 2]-rois[:, 0])*(rois[:, 3]-rois[:, 1])
    area2 = (bboxes[:, 2]-bboxes[:, 0])*(bboxes[:, 3]-bboxes[:, 1])
    union = area1[:, None]+area2[None, :]-intersection
    
    return intersection/(union+epsilon)

def annotate_rois(rois, bboxes, ids, cat_thresh=0.3):
    ious = get_ious(rois.boxes, bboxes)
    max_ious, max_idxs = ious.max(dim=1)
    valid_mask = max_ious>cat_thresh
    rois.ids[valid_mask] = ids[max_idxs[valid_mask]]
    offsets = bboxes[max_idxs]-rois.boxes
    widths, heights = rois.boxes[:, 2], rois.boxes[:, 3]
    rois.offsets = torch.stack([
        offsets[:, 0] / widths,  # dx / width
        offsets[:, 1] / heights,  # dy / height
        offsets[:, 2] / widths,  # dw / width
        offsets[:, 3] / heights  # dh / height
    ], dim=1)
    rois.ious = max_ious
    return rois

class ObjectDataset(Dataset):
    def __init__(self, df, extract_rois, cat_thresh=0.3, crop_size=(224,224)):
        self.df = df.groupby('image_id').agg(list).reset_index()
        self.extract_rois = extract_rois
        self.crop_size = crop_size
        self.cat_thresh = cat_thresh
        self.img_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path, bboxes, ids = agg_df.loc[idx]
        img = Image.open(img_path)
        img = self.img_tfms(img)
        bboxes = torch.tensor(bboxes, dtype=torch.float16)
        ids = torch.tensor(ids, dtype=torch.long)

        rois = self.extract_rois(img)
        rois = annotate_rois(rois, bboxes, ids, self.cat_thresh)
        crops = []
        for box in rois.boxes:
            x_min, y_min, w, h = box.int().tolist()
            crop_img = crop(img, y_min, x_min, h, w)
            crops.append(crop_img)
        crops = [resize(crop_img, self.crop_size) for crop_img in crops]
        crops = torch.stack(crops)

        return {'crops':crops, 'rois':rois}

    def collate_fn(self, batch):
        all_crops = torch.cat([item['crops'] for item in batch], dim=0)
        all_ids = torch.cat([item['rois'].ids for item in batch], dim=0)
        all_offsets = torch.cat([item['rois'].offsets for item in batch], dim=0)
        
        return {'crops':all_crops, 'ids':all_ids, 'offsets':all_offsets}