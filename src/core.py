import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class JSONLoader:
    def __init__(self, data_folder):
        data_folder = Path(data_folder)
        self.ann_folder = data_folder/'annotations'
        self.img_folder = data_folder/'images'

    def load_train(self):
        with open(self.ann_folder/'train.json', 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data['annotations'])
        id2img = {d['id']:self.img_folder/d['file_name'] for d in data['images']}
        cat_id2id = {d['id']:i+1 for i, d in enumerate(data['categories'])}
        id2label = {i+1:d['name'] for i, d in enumerate(data['categories'])}
        id2label[0] = 'background'

        df.category_id = df.category_id.replace(cat_id2id)
        df = df.groupby('image_id').agg(list).reset_index()
        return df, id2label, id2img

    def load_test(self, id2img):
        with open(self.ann_folder/'test.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data['annotations'])
        df = df.groupby('image_id').agg(list).reset_index()
        id2img.extend({d['id']:self.img_folder/d['file_name'] for d in data['images']})
        return df

def show_bbs(img, bboxes, ids=None, id2label=None):
    draw = ImageDraw.Draw(img)
    if ids:
        labels = [id2label[id] for id in ids]
    for i, bbox in enumerate(bboxes):
        x_min, y_min, w, h = bbox
        draw.rectangle(((x_min, y_min), (x_min+w, y_min+h)), outline="green", width=3)
        if ids: 
            draw.text((x_min, y_min-10), labels[i], fill="white")
    plt.imshow(img)
    plt.axis('off')
    plt.show()