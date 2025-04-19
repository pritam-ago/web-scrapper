import torch
from model import CaptchaModel

# Load the model
model = CaptchaModel(num_classes=62, max_length=5)  # Adjust num_classes according to your charset
model.load_state_dict(torch.load('captcha_model.pth'))
model.eval()

# Example for a single image prediction
from PIL import Image
import torchvision.transforms as transforms

image_path = "test_image.png"  # Path to the test image
image = Image.open(image_path).convert("L")

# Preprocessing
transform = transforms.Compose([transforms.Resize((64, 256)), transforms.ToTensor()])
image = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(image.cuda())
    output = output.argmax(2)  # Get the indices of the predicted classes

# Convert indices to characters
idx_to_char = {idx: char for idx, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-")}
predicted_label = ''.join([idx_to_char[i.item()] for i in output[0]])

print(f"Predicted Label: {predicted_label}")
