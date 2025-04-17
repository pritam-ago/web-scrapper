import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CaptchaDataset(Dataset):
    def __init__(self, csv_file, image_dir, max_length=5, charset="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.max_length = max_length
        self.charset = charset
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.idx_to_char = {i: c for i, c in enumerate(self.charset)}
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((50, 200)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['filename']))
        image = self.transform(image)
        label = row['label'].upper().ljust(self.max_length, '-')
        label_tensor = [self.char_to_idx[c] for c in label]
        return image, label_tensor
