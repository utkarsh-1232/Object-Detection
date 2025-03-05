from typing import Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import box_iou
from torchvision.transforms.functional import resized_crop

@dataclass
class ROIs:
    boxes: torch.FloatTensor
    ids: Optional[torch.IntTensor] = None
    offsets: Optional[torch.FloatTensor] = None
    ious: Optional[torch.FloatTensor] = None
    
    def __post_init__(self):
        self.ids = torch.full((self.boxes.shape[0],), 0, dtype=torch.long)

def annotate_rois(rois, bboxes, ids, cat_thresh=0.3):
    rois.boxes = torch.cat([rois.boxes[:,:2], rois.boxes[:,:2]+rois.boxes[:,2:]], dim=1)
    bboxes = torch.cat([bboxes[:,:2], bboxes[:,:2]+bboxes[:,2:]], dim=1)
    ious = box_iou(rois.boxes, bboxes)
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
    def __init__(self, df, extract_rois, cat_thresh=0.3, crop_size=(112,112)):
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
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        ids = torch.tensor(ids, dtype=torch.long)

        rois = self.extract_rois(img)
        rois = annotate_rois(rois, bboxes, ids, self.cat_thresh)
        crops = []
        for box in rois.boxes:
            x_min, y_min, w, h = box.int().tolist()
            cropped_img = resized_crop(img, top=y_min, left=x_min, height=h, width=w, size=self.crop_size)
            crops.append(cropped_img)
        crops = torch.stack(crops)

        return {'crops':crops, 'rois':rois}

    def collate_fn(self, batch):
        all_crops = torch.cat([item['crops'] for item in batch], dim=0)
        all_ids = torch.cat([item['rois'].ids for item in batch], dim=0)
        all_offsets = torch.cat([item['rois'].offsets for item in batch], dim=0)
        
        return {'crops':all_crops, 'ids':all_ids, 'offsets':all_offsets}