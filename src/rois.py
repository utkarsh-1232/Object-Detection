import torch
import torchvision.io as io
from torchvision.ops import box_iou
from selectivesearch import selective_search
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def extract_rois(img, scale=200, min_size=100):
    img_np = img.permute(1, 2, 0).numpy()
    h,w = img_np.shape[:2]; img_area = h*w
    _, regions = selective_search(img_np, scale=scale, min_size=min_size)
    unique_rois = set()
    for r in regions:
        rect, size = r['rect'], r['size']
        if (size>0.05*img_area) and (size<img_area) and (rect not in unique_rois):
            unique_rois.add(rect)
    rois = torch.empty((0, 4))
    if len(unique_rois)>0:
        rois = torch.tensor([list(x) for x in unique_rois], dtype=torch.float32)
    return rois

def calculate_offsets(rois, gtbbs):
    # Convert to center coordinates
    roi_centers = rois[:, :2] + rois[:, 2:]/2
    gt_centers = gtbbs[:, :2] + gtbbs[:, 2:]/2
    
    # Calculate offsets with numerical stability
    epsilon = 1e-8
    roi_widths = rois[:, 2] + epsilon
    roi_heights = rois[:, 3] + epsilon
    
    dx = (gt_centers[:, 0] - roi_centers[:, 0]) / roi_widths
    dy = (gt_centers[:, 1] - roi_centers[:, 1]) / roi_heights
    dw = torch.log(gtbbs[:, 2] / roi_widths)
    dh = torch.log(gtbbs[:, 3] / roi_heights)
    
    return torch.stack([dx, dy, dw, dh], dim=1)

def apply_offsets(rois, pred_offsets):
    # Convert ROIs to center coordinates
    centers = rois[:, :2] + rois[:, 2:]/2
    widths = rois[:, 2]
    heights = rois[:, 3]
    
    # Unpack offsets
    dx = pred_offsets[:, 0]
    dy = pred_offsets[:, 1]
    dw = pred_offsets[:, 2]
    dh = pred_offsets[:, 3]
    
    # Apply transformations
    refined_centers_x = centers[:, 0] + dx * widths
    refined_centers_y = centers[:, 1] + dy * heights
    refined_widths = widths * torch.exp(dw)
    refined_heights = heights * torch.exp(dh)
    
    # Convert back to [x, y, w, h] format
    refined_boxes = torch.stack([
        refined_centers_x - refined_widths/2,
        refined_centers_y - refined_heights/2,
        refined_widths,
        refined_heights
    ], dim=1)
    
    return refined_boxes

def annotate_rois(rois, gtbbs, cat_ids, cat_thresh=0.3):
    r_xyXY = torch.cat([rois[:,:2], rois[:,:2]+rois[:,2:]-1], dim=1)
    gt_xyXY = torch.cat([gtbbs[:,:2], gtbbs[:,:2]+gtbbs[:,2:]-1], dim=1)
    ious = box_iou(r_xyXY, gt_xyXY)
    max_ious, max_idxs = ious.max(dim=1)
    
    valid_mask = max_ious>cat_thresh
    roi_ids = torch.full((rois.shape[0],), 0, dtype=torch.long)
    roi_ids[valid_mask] = cat_ids[max_idxs[valid_mask]]
    offsets = calculate_offsets(rois, gtbbs[max_idxs])
    return roi_ids, offsets, max_ious

def get_annotated_rois(row, id2img, scale=200, min_size=100, cat_thresh=0.3):
    img_id, gtbbs, cat_ids = row[['image_id','bbox','category_id']]
    img = io.read_image(id2img[img_id], mode=io.ImageReadMode.RGB)
    img_id = torch.tensor(img_id, dtype=torch.long)
    gtbbs = torch.tensor(gtbbs, dtype=torch.float32)
    cat_ids = torch.tensor(cat_ids, dtype=torch.long)

    rois = extract_rois(img, scale=scale, min_size=min_size)
    roi_ids, offsets, max_ious = annotate_rois(rois, gtbbs, cat_ids, cat_thresh)
    return img_id, rois, roi_ids, offsets