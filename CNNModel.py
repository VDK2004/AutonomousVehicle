import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from fastai.vision.all import *
from sklearn.model_selection import train_test_split

# Parameters
IMG_HEIGHT, IMG_WIDTH = 100, 100
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Data augmentation for FastAI
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),
])

# Dataset class for in-memory data
class TrackmaniaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image to uint8 type for PIL compatibility
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert to tensor and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to CxHxW format
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# Function to load data from directory
def load_data(data_dir):
    images = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        for img_file in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, img_file))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(int(label_dir))  # label 0 for straight, 1 for left, 2 for right, 3 for backward
    return np.array(images), np.array(labels)

# Load and preprocess the data
data_dir = "trackmania_data"
X, y = load_data(data_dir)
X = X / 255.0  # Normalize images

# Handle any invalid labels outside expected range
if np.any((y < 0) | (y > 3)):
    raise ValueError("Some labels are outside the expected range 0-3.")

# Balance the data through oversampling
oversampled_images = []
oversampled_labels = []
for image, label in zip(X, y):
    if label in [1, 2]:  # Oversample underrepresented classes (left and right)
        for _ in range(5):  # Duplicate examples
            oversampled_images.append(image)
            oversampled_labels.append(label)
    else:
        oversampled_images.append(image)
        oversampled_labels.append(label)

X_balanced = np.array(oversampled_images)
y_balanced = np.array(oversampled_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Create dataset objects
train_dataset = TrackmaniaDataset(X_train, y_train, transform=transform)
test_dataset = TrackmaniaDataset(X_test, y_test, transform=transforms.Resize((IMG_HEIGHT, IMG_WIDTH)))

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Wrap DataLoader with FastAI's DataLoaders
dls = DataLoaders(train_loader, test_loader)

# CNN Model definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 25 * 25, 128)  # Adjust based on your input size
        self.fc2 = nn.Linear(128, 4)  # 4 classes: forward, left, right, backward

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model
model = CNNModel()

# Create a learner using FastAI's Learner
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), opt_func=Adam, metrics=accuracy)

# Train the model
learn.fit_one_cycle(EPOCHS, LEARNING_RATE)

# Save the trained model
learn.save('trackmania_model')

# Evaluate the model on the test set
learn.validate()
