from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CandlestickDataset(Dataset):
    def __init__(self, csv_path, transform=None, img_dir=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        if self.img_dir:
            import os
            p = os.path.join(self.img_dir, path) if not os.path.isabs(path) else path
        else:
            p = path
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = row.get('label', None)
        return img, label
