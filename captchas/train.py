import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CaptchaDataset
from model import CaptchaCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 15
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
MAX_LEN = 5


dataset = CaptchaDataset("labels.csv", "captchas", max_length=MAX_LEN, charset=CHARSET)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = CaptchaCNN(num_chars=MAX_LEN, num_classes=len(CHARSET)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def accuracy(preds, labels):
    preds = preds.argmax(2)
    correct = 0
    for p, l in zip(preds, labels):
        if all(p[i] == l[i] for i in range(len(l))):
            correct += 1
    return correct / len(labels)

for epoch in range(EPOCHS):
    model.train()
    total_loss, acc = 0, 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = torch.tensor(labels).to(device)

        out = model(images)
        loss = sum(F.cross_entropy(out[:, i, :], labels[:, i]) for i in range(MAX_LEN))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc += accuracy(out.detach().cpu(), labels.cpu())

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Acc: {acc/len(dataloader):.2%}")

torch.save(model.state_dict(), "captcha_model.pth")
print("✅ Model saved as captcha_model.pth")
