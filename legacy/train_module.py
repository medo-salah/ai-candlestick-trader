import os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from .dataset import CandlestickDataset
from .model import create_model

def get_transforms(img_size=224):
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

def train(train_csv='data/train.csv', img_dir=None, epochs=1, batch_size=16, img_size=224, model_name='resnet18', save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    transforms = get_transforms(img_size)
    ds = CandlestickDataset(train_csv, transform=transforms, img_dir=img_dir)
    # map labels to indices
    labels = sorted(list(set(ds.df['label'].tolist())))
    label_to_idx = {l:i for i,l in enumerate(labels)}
    # build numeric label column
    ds.df['__label_idx'] = ds.df['label'].map(label_to_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name, num_classes=len(labels), pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for imgs, labels_batch in loader:
            # labels_batch are strings; convert to indices
            y = [label_to_idx[l] for l in labels_batch]
            y = torch.tensor(y, dtype=torch.long).to(device)
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
        avg = running / n if n else 0.0
        print(f"Epoch {epoch+1}/{epochs} - loss {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            path = os.path.join(save_dir, 'best_model.pth')
            torch.save({'model_state_dict': model.state_dict(), 'labels': labels}, path)
            print('Saved model to', path)
