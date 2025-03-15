import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_bbs(img_path, bboxes, labels=None):
    img = plt.imread(img_path)
    _, ax = plt.subplots(1)
    ax.imshow(img)
    for i, bbox in enumerate(bboxes):
        xmin, ymin, w, h = bbox
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        if labels:
            ax.text(xmin+5, ymin+10, labels[i], color='black', fontsize=8, backgroundcolor='white')
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

def load_data(data_folder, split='train'):
    data_folder = Path(data_folder)
    data = json.load((data_folder/f'{split}.json').open())

    id2label = {d['id']:d['name'] for d in data['categories']}
    id2img = {d['id']:data_folder/f'train/{d['file_name']}' for d in data['images']}

    df = pd.DataFrame(data['annotations'])
    df = df.query('ignore!=1').drop(columns=['segmentation','ignore'])
    agg_df = df.groupby('image_id').agg(list).reset_index()
    return agg_df, id2label, id2img