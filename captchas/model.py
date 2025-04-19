import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

transform = transforms.Compose([
    transforms.Resize((64, 256)),  # Resize image to fixed dimensions
    transforms.ToTensor(),         # Convert image to a tensor
])

class CaptchaDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, max_length=5, charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"):
        """
        Args:
            csv_file (string): Path to the CSV file with filenames and labels.
            image_folder (string): Path to the folder with CAPTCHA images.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_length (int): Maximum length of CAPTCHA strings.
            charset (str): The set of characters in the CAPTCHA images.
        """
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

        # Load the image and convert to grayscale
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # Pad label if it's shorter than max_length
        label = label.ljust(self.max_length, '-')

        try:
            label_tensor = [self.char_to_idx[c] for c in label]
        except KeyError as e:
            print(f"Bad character '{e}' in label '{label}' (idx {idx})")
            raise

        return image, torch.tensor(label_tensor, dtype=torch.long)

class CaptchaModel(nn.Module):
    def __init__(self, num_classes, max_length):
        super(CaptchaModel, self).__init__()

        self.max_length = max_length
        self.num_classes = num_classes

        # CNN Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 64, 1024)
        self.fc2 = nn.Linear(1024, num_classes * max_length)

    def forward(self, x):
        # Apply convolutions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten
        x = x.view(-1, 256 * 16 * 64)
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)
        x = x.view(-1, self.max_length, self.num_classes)
        
        return x
