import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision

# Data Path and Transformations
data_dir = 'Variational_Autoencoders_VAEs/data'  # Path to the 'data' folder
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading the Dataset
dataset = ImageFolder(root=data_dir, transform=transform)

# Splitting the Dataset
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Emotion Classes
class_names = dataset.classes
num_classes = len(class_names)

# Visualizing the Data (an example batch)
import matplotlib.pyplot as plt
import numpy as np

def imshow(image):
    image = image / 2 + 0.5
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get the first batch and visualize it
dataiter = iter(dataloader)
images, labels = dataiter.__next__()

# Visualize (If you want to see the images, uncomment the line below)
# imshow(torchvision.utils.make_grid(images))
# print(' '.join(f'{class_names[labels[j]]:5s}' for j in range(batch_size)))

# Defining the VAE Model

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(32 * 16 * 16, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )



    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        # Encoding
        hidden = self.encoder(x)
        hidden = self.fc1(hidden)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        # Print the shapes of mu, logvar, and hidden for debugging
        # If you want to see the shapes, uncomment the lines below
        # print("mu shape:", mu.shape)
        # print("logvar shape:", logvar.shape)
        # print("hidden shape:", hidden.shape)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decoding
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


# Hyperparameters
input_dim = (3, 64, 64)  # Input image dimensions (channels, height, width)
hidden_dim = 256
latent_dim = 20

# Create VAE model
vae_model = VAE(input_dim, hidden_dim, latent_dim)

# Training the VAE Model

# Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae_model(data)

            # Calculate reconstruction loss and KL divergence
            reconstruction_loss = criterion(reconstructed, data)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total loss
            loss = reconstruction_loss + kl_divergence

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print("Training Finished")

    # Save the model
    torch.save(vae_model.state_dict(), 'vae_model.pth')
    print("Model Saved")