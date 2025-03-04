from src.core import *
from typing import Optional
from dataclasses import dataclass
import torch

@dataclass
class ROIs:
    boxes: torch.FloatTensor
    categories: Optional[torch.IntTensor] = None
    offsets: Optional[torch.FloatTensor] = None
    ious: Optional[torch.FloatTensor] = None
    
    def __post_init__(self):
        self.categories = torch.full((self.boxes.shape[0],), 0, dtype=torch.long)

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

def annotate_rois(bboxes, ids, rois, cat_thresh=0.3):
    ious = get_ious(rois.boxes, bboxes)
    max_ious, max_idxs = ious.max(dim=1)
    valid_mask = max_ious>cat_thresh
    rois.categories[valid_mask] = ids[max_idxs[valid_mask]]
    rois.offsets = bboxes[max_idxs]-rois.boxes
    rois.ious = max_ious
    return rois