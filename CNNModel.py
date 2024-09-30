import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Parameters
IMG_HEIGHT, IMG_WIDTH = 100, 100  # Rescale images
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Dataset class for PyTorch
class TrackmaniaDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Reorder dimensions for PyTorch
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# Functie om data te laden
def load_data(data_dir):
    images = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        for img_file in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, img_file))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(int(label_dir))  # label 0 voor recht, 1 voor links, 2 voor rechts
    return np.array(images), np.array(labels)

# Laad en split de data
data_dir = "trackmania_data"
X, y = load_data(data_dir)
X = X / 255.0  # Normaliseer beelden

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TrackmaniaDataset(X_train, y_train)
test_dataset = TrackmaniaDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        # Aangepast voor de juiste invoergrootte
        self.fc1 = nn.Linear(64 * 23 * 23, 128)  # 64 kanalen * 23 hoogte * 23 breedte
        self.fc2 = nn.Linear(128, 3)  # 3 outputs: z, q, d

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        print(x.shape)  # Print de vorm van de tensor na de convoluties en pooling

        x = x.view(x.size(0), -1)  # Flatten de tensor op basis van batch size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Initialiseer model, optimizer, en loss-functie
model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

    # Opslaan van het model
    torch.save(model.state_dict(), 'trackmania_model.pth')

train_model()
