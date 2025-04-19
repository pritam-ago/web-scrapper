import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CaptchaDataset, CaptchaModel
from torchvision import transforms

print(torch.cuda.is_available())
# Parameters
csv_file = 'labels.csv'  # Path to the CSV file
image_folder = 'captchas/'  # Path to the images
max_length = 5
charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"

# Create Dataset and DataLoader
transform = transforms.Compose([transforms.Resize((64, 256)), transforms.ToTensor()])
dataset = CaptchaDataset(csv_file=csv_file, image_folder=image_folder, transform=transform, max_length=max_length, charset=charset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model, loss function, and optimizer
num_classes = len(charset)
model = CaptchaModel(num_classes=num_classes, max_length=max_length)
model = model.cuda()

criterion = nn.CTCLoss(blank=num_classes-1)  # CTC Loss requires a "blank" index, usually the last index
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.squeeze(1)  # Remove single channel dimension
        images, labels = images.cuda(), labels.cuda()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate CTC loss
        input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).cuda()
        target_lengths = torch.full((images.size(0),), labels.size(1), dtype=torch.long).cuda()

        loss = criterion(outputs.log_softmax(2), labels, input_lengths, target_lengths)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save the model after training
torch.save(model.state_dict(), 'captcha_model.pth')
