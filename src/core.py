import io, sys, json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def show_bbs(img_path, bboxes, labels=None, ax=None):
    img = plt.imread(img_path)
    if ax is None: _, ax = plt.subplots(1)
    ax.imshow(img)
    for i, bbox in enumerate(bboxes):
        xmin, ymin, w, h = bbox
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        if labels:
            ax.text(xmin+5, ymin+10, labels[i], color='black', fontsize=8, backgroundcolor='white')
    ax.axis('off')  # Hide axes for better visualization

def show_imgs(ann_file, nrows, id2label, id2img):
    data = json.load(open(ann_file))
    if type(data)==dict: data = data['annotations']
    df = pd.DataFrame(data).groupby('image_id').agg(list)
    sample = df.sample(nrows*3)
    
    _, axes = plt.subplots(nrows, 3, figsize=(15, 5*nrows))
    for ax, (img_id, row) in zip(axes.flatten(), sample.iterrows()):
        img_path = id2img[img_id]
        cat_ids, bboxes = row[['category_id','bbox']]
        titles = [id2label[id] if id!=0 else 'bg' for id in cat_ids]
        if 'score' in row:
            scores = row['score']
            titles = [f'{l}, {score:.2f}' for l, score in zip(titles, scores)]
        show_bbs(img_path, bboxes, titles, ax)
        
    plt.tight_layout()
    plt.show()

def load_data(ann_file, imgs_folder):
    data = json.load(open(ann_file))

    id2label = {d['id']:d['name'] for d in data['categories']}
    id2img = {d['id']:f"{imgs_folder}/{d['file_name']}" for d in data['images']}

    df = pd.DataFrame(data['annotations'])
    df = df.query('ignore!=1').drop(columns=['segmentation','ignore'])
    agg_df = df.groupby('image_id').agg(list).reset_index()
    return agg_df, id2label, id2img

def calc_mAP(gt_path, pred_path):
    with io.StringIO() as buf:
        save_stdout = sys.stdout
        sys.stdout = buf  # Redirect standard output
        coco_gt = COCO(gt_path)
        coco_pred = coco_gt.loadRes(pred_path)
        cocoEval = COCOeval(coco_gt, coco_pred, 'bbox')
        cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
        sys.stdout = save_stdout  # Restore standard output
    return cocoEval.stats[0]