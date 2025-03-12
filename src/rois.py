from PIL import Image
import torch
from torchvision import transforms
from torchvision.ops import box_iou
from selectivesearch import selective_search

def extract_rois(img):
    img_area = img.shape[1]*img.shape[2]
    img_int = (img*255).to(torch.uint8).permute((1,2,0))
    _, regions = selective_search(img_int, scale=200, min_size=100)
    rois = torch.tensor([r['rect'] for r in regions])
    sizes = torch.tensor([r['size'] for r in regions])
    mask = (sizes>0.05*img_area) & (sizes<img_area)
    return rois[mask, :]

def annotate_rois(rois, gtbbs, ids, cat_thresh=0.3):
    r_xyXY = torch.cat([rois[:,:2], rois[:,:2]+rois[:,2:]], dim=1)
    gt_xyXY = torch.cat([gtbbs[:,:2], gtbbs[:,:2]+gtbbs[:,2:]], dim=1)
    ious = box_iou(r_xyXY, gt_xyXY)
    max_ious, max_idxs = ious.max(dim=1)
    
    valid_mask = max_ious>cat_thresh
    roi_ids = torch.full((rois.shape[0],), 0, dtype=torch.long)
    roi_ids[valid_mask] = ids[max_idxs[valid_mask]]
    offsets = gtbbs[max_idxs]-rois
    return roi_ids, offsets

def get_annotated_rois(row, id2img, cat_thresh=0.3):
    img_id, gtbbs, ids = row
    img = Image.open(id2img[img_id]).convert('RGB')
    img_tensor = transforms.ToTensor()(img)
    _, H, W = img_tensor.shape
    img_id = torch.tensor(img_id, dtype=torch.long)
    gtbbs = torch.tensor(gtbbs, dtype=torch.float32)
    ids = torch.tensor(ids, dtype=torch.long)

    rois = extract_rois(img_tensor)
    roi_ids, offsets = annotate_rois(rois, gtbbs, ids, cat_thresh)
    img_ids = img_id.repeat(rois.shape[0])
    img_dims = torch.tensor([W,H,W,H])[None,:].repeat(rois.shape[0], 1)
    return img_ids, rois, roi_ids, offsets, img_dims