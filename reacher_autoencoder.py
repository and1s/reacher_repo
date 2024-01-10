import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Load dataset from pkl file
with open('reacher_dataset.pkl', 'rb') as f:
    data_pairs = pickle.load(f)

# Get images and normalize
dataset = np.array([pair[0] for pair in data_pairs])

dataset = dataset.transpose((0, 3, 1, 2))

# Normalize images
dataset = dataset.astype(np.float32) / 255.0

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Convert to PyTorch tensor
        return torch.from_numpy(self.dataset[index])

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(CustomDataset(dataset), [train_size, validation_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)


# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 8*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (8, 16, 16)),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the model, loss function, and optimizer
model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
print('Training...')
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    # Training phase
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # Validation phase
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for data in validation_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            validation_loss += loss.item()

    # Calculate and print average losses
    train_loss /= len(train_loader.dataset)
    validation_loss /= len(validation_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.10f}, Validation Loss: {validation_loss:.10f}')

# Save the model
torch.save(model.state_dict(), 'reacher_autoencoder.pt')
print('Model saved to reacher_autoencoder.pt')

# test model on random image from validation set and save image to a png
import matplotlib.pyplot as plt

# Get a random image from the validation set
image = validation_dataset[np.random.randint(0, len(validation_dataset))]
image = image.unsqueeze(0).to(device)

# Get the autoencoder output
model.eval()
with torch.no_grad():
    output = model(image)

# Convert to numpy
image = image.cpu().numpy()
output = output.cpu().numpy()

# take images from 0-1 to 0-255
image = image * 255
output = output * 255

image = image.astype(np.uint8)
output = output.astype(np.uint8)


# save to jpg
plt.imsave('reacher_autoencoder.jpg', output[0].transpose(1, 2, 0))
print('Image saved to reacher_autoencoder.jpg')

# save image to jpg
plt.imsave('reacher_autoencoder_original.jpg', image[0].transpose(1, 2, 0))
print('Image saved to reacher_autoencoder_original.jpg')

