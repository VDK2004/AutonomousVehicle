import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import this for functional operations
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image


# Parameters
IMG_HEIGHT, IMG_WIDTH = 100, 100  # Rescale images
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Rotatie tot 15 graden
    transforms.RandomHorizontalFlip(),      # Horizontaal spiegelen
    transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),  # Rescale
    transforms.ToTensor()  # Verander in tensor
])

# Dataset class for PyTorch
# Dataset class for PyTorch
# Dataset class for PyTorch
class TrackmaniaDataset(Dataset):
    def __init__(self, images, labels, augment=False, transform=None):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Zorg ervoor dat de afbeelding een uint8 type heeft
        image = (image * 255).astype(np.uint8)  # Zorg dat het datatype uint8 is

        # Converteer naar een PIL-afbeelding voor augmentaties
        image = Image.fromarray(image)

        # Apply data augmentation als aangegeven
        if self.augment and self.transform is not None:
            image = self.transform(image)

        # Converteer terug naar tensor
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize naar [0, 1]

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
# Laad en controleer de data
X, y = load_data(data_dir)
X = X / 255.0  # Normaliseer beelden

# Controleer de labels
print("Unieke labels in de dataset:", np.unique(y))

# Controleer of de labels binnen het bereik 0-2 liggen
if np.any((y < 0) | (y > 2)):
    print("Er zijn labels buiten het verwachte bereik 0-2.")
    # Optioneel: je kunt de foutieve data hier filteren


# Balancing through oversampling
oversampled_images = []
oversampled_labels = []

for image, label in zip(X, y):
    if label in [1, 2]:  # Oversample underrepresented classes (left and right)
        for _ in range(5):  # Duplicate examples 5 times
            oversampled_images.append(image)
            oversampled_labels.append(label)
    else:
        oversampled_images.append(image)
        oversampled_labels.append(label)

X_balanced = np.array(oversampled_images)
y_balanced = np.array(oversampled_labels)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

train_dataset = TrackmaniaDataset(X_train, y_train, augment=True)
test_dataset = TrackmaniaDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define your CNN layers (convolutions, pooling, etc.)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 25 * 25, 128)  # Adjust based on your input size
        self.fc2 = nn.Linear(128, 4)  # Change to 4 classes: forward, left, right, backward

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
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
            # Verplaats naar het juiste apparaat als je CUDA gebruikt
            images, labels = images.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            # Model output
            outputs = model(images)  # outputs heeft vorm [batch_size, num_classes]

           

            # Loss berekenen
            loss = criterion(outputs, labels)

            # Backpropagation en update
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

    # Model opslaan
    torch.save(model.state_dict(), 'trackmania_model.pth')


train_model()

# Evaluatie
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

evaluate_model()
