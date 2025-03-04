import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from torch import tensor
from torch.utils.data import Dataset

# with open('data/annotations/train.json', 'r') as f:
#     data = json.load(f)

# id2img = {d['id']:f'data/train_images/{d['file_name']}' for d in data['images']}
# cat_id2id = {d['id']:i for i, d in enumerate(data['categories'])}
# id2label = {i:d['name'] for i, d in enumerate(data['categories'])}

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
        draw = ImageDraw.Draw(img)
        labels = [id2label[id] for id in ids]
        
        for label, box in zip(labels, boxes):
            x_min, y_min, w, h = box
            draw.rectangle(((x_min, y_min), (x_min+w, y_min+h)), outline="green", width=3)
            draw.text((x_min, y_min-10), label, fill="white")
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def get_iou(boxA, boxB, epsilon=1e-5):
    x_min = max(boxA[0], boxB[0])
    y_min = max(boxA[1], boxB[1])
    x_max = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    y_max = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    w, h = x_max-x_min, y_max-y_min

    if w<0 or h<0: return 0
    intersection = w*h
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - intersection
    return intersection/(union+epsilon)

from pathlib import Path
path = Path('../data/annotations/train.json')
print(path.exists())