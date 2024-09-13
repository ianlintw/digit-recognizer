#importing PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset

#Importing libraries related to data and viz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Load and prepare data
# Make sure the files are in the same directory as your script, or
train = pd.read_csv('train.csv') 
test = pd.read_csv('test.csv')
X_train = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train['label'].values

class DigitDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2)  # Change the order of dimensions
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        image = self.X[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.y[idx]

transform = transforms.Compose([
    transforms.Normalize([0.5], [0.5])
])

dataset = DigitDataset(X_train, y_train, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#CNN Classsifer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Ensure x is the correct shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if it's missing
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
cnnclassifier = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnnclassifier.parameters(), lr=0.001)

#Training loop
num_epochs = 10
for epoch in range(num_epochs):
    cnnclassifier.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnnclassifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
        
#Evaluate the model
cnnclassifier.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnnclassifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

#Save the model
torch.save(cnnclassifier.state_dict(), 'cnnclassifier.pth')

#Load the model
