import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import os

# ---------------- Preprocessing ---------------- #
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---------------- Custom Dataset ---------------- #
class SheepDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.imgs = glob.glob(os.path.join(img_dir, "*.jpg"))  # or *.png
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        label = 1  # only one class: sheep → positive
        if self.transform:
            img = self.transform(img)
        return img, label

# Dataset paths (point to IMAGES, not labels)
train_dir = "COCO/images/train2017/"
val_dir = "COCO/images/val2017/"

train_data = SheepDataset(train_dir, transform=transform)
val_data = SheepDataset(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)

# ---------------- Model ---------------- #
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)  # single output for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

# ---------------- Training Setup ---------------- #
criterion = nn.BCEWithLogitsLoss()   # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------- Training Loop ---------------- #
def train_model(epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)  # [batch]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            total += labels.size(0)
            correct += (preds == labels.long()).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Train Acc: {acc:.2f}%")
        validate()

# ---------------- Validation Loop ---------------- #
def validate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            total += labels.size(0)
            correct += (preds == labels.long()).sum().item()

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# ---------------- Main ---------------- #
if __name__ == "__main__":
    train_model(epochs=10)
    torch.save(model.state_dict(), "sheep_behavior_model.pth")
    print("✅ Model saved as sheep_behavior_model.pth")
