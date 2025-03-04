import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from torch import tensor
from torch.utils.data import Dataset

path = Path('data')
with open(path/'annotations/train.json', 'r') as f:
    data = json.load(f)

id2img = {d['id']:f'{path}/train_images/{d['file_name']}' for d in data['images']}
cat_id2id = {d['id']:i+1 for i, d in enumerate(data['categories'])}
id2label = {i+1:d['name'] for i, d in enumerate(data['categories'])}
id2label[0] = 'background'

def show_bbs(img, boxes, labels=None):
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        x_min, y_min, w, h = box
        draw.rectangle(((x_min, y_min), (x_min+w, y_min+h)), outline="green", width=3)
        if labels:
            draw.text((x_min, y_min-10), labels[i], fill="white")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

class ObjectDataset(Dataset):
    def __init__(self, df, tfms=None):
        self.df = df.groupby('image_id').agg(list).reset_index()
        self.tfms = tfms

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_id, boxes, ids = [*self.df.loc[idx]]
        img = Image.open(id2img[img_id])
        if self.tfms:
            img = self.tfms(img)
        boxes, ids = tensor(boxes), tensor(ids)
        return img, boxes, ids

    def show(self, idx):
        img_id, boxes, ids = [*self.df.loc[idx]]
        img = Image.open(id2img[img_id])
        labels = [id2label[id] for id in ids]
        show_bbs(img, boxes, labels)