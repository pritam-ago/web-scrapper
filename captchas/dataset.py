import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class CaptchaDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, max_length=5, charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.max_length = max_length
        self.charset = charset

        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        print("Charset loaded ✅")
        print("char_to_idx:", self.char_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, row['filename'])
        label = row['label']

        image = Image.open(image_path).convert("L")  # grayscale

        # Apply transformation to the image
        if self.transform:
            image = self.transform(image)

        # pad label if shorter than max_length
        label = label.ljust(self.max_length, '-')

        try:
            label_tensor = [self.char_to_idx[c] for c in label]
        except KeyError as e:
            print(f"Bad character '{e}' in label '{label}' (idx {idx})")
            raise

        return image, torch.tensor(label_tensor, dtype=torch.long)

# Define your transforms (including ToTensor())
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
])

# Assuming you are using a DataLoader
dataset = CaptchaDataset(csv_file='path_to_csv.csv', image_folder='path_to_images', transform=transform)
