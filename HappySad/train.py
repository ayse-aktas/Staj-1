import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

# Cihaz kontrolü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Veri setini yükle
df = pd.read_csv("fer2013_dataset/fer2013.csv")
X = df['pixels'].tolist()
y = df['emotion'].values

# Eğitim ve doğrulama verisini ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Veri kümesi sınıfı
class FER2013Dataset(Dataset):
    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        img = np.fromstring(self.pixels[idx], dtype=np.uint8, sep=' ').reshape(48, 48)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Veri artırımı ve transformlar
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset ve loader'lar
train_dataset = FER2013Dataset(X_train, y_train, train_transform)
val_dataset = FER2013Dataset(X_val, y_val, val_transform)

# Sınıf ağırlıkları hesapla (dengesizlik için)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Sampler (alternatif olarak kullanılabilir)
# class_sample_counts = np.bincount(y_train)
# weights = 1. / class_sample_counts[y_train]
# sampler = WeightedRandomSampler(weights, len(weights))
# train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model sınıfı (daha derin CNN)
class CNNEmotionModel(nn.Module):
    def __init__(self):
        super(CNNEmotionModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Model oluştur
model = CNNEmotionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim
epochs = 30
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Doğrulama
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Validation Accuracy: {acc:.2f}%")

    # En iyi modeli kaydet
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "model.pth")
        print(f"✅ Yeni en iyi model kaydedildi: {acc:.2f}%")

print(f"🎉 Eğitim tamamlandı. En iyi doğruluk: {best_acc:.2f}%")
