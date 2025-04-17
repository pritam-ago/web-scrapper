import torch.nn as nn

class CaptchaCNN(nn.Module):
    def __init__(self, num_chars=5, num_classes=36):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 25, 1024), nn.ReLU(),
            nn.Linear(1024, num_chars * num_classes)
        )
        self.num_chars = num_chars
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.view(-1, self.num_chars, self.num_classes)
