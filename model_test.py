import torch
import matplotlib.pyplot as plt
from main import VAE, dataloader  # Import your dataloader
import numpy as np

# Hyperparameters
input_dim = (1, 64, 64)  # Input image dimensions (channels, height, width)
hidden_dim = 256
latent_dim = 20

# Create an instance of your VAE model
vae_model = VAE(input_dim, hidden_dim, latent_dim)

# Load the trained model state
vae_model.load_state_dict(torch.load('vae_model.pth'))
vae_model.eval()

def plot_images(images, title):
    images = images / 2 + 0.5  # Undo the normalization
    np_images = images[0, 0].cpu().numpy()  # Remove channel dimension and convert to numpy
    plt.figure(figsize=(10, 5))
    plt.imshow(np_images, cmap='gray')  # Use grayscale colormap
    plt.title(title)
    plt.axis('off')
    plt.show()

# Image Reconstruction
with torch.no_grad():
    images, _ = next(iter(dataloader))
    reconstructed, _, _ = vae_model(images)
    
plot_images(images, title='Original Images')
plot_images(reconstructed, title='Reconstructed Images')

# Generating New Samples
with torch.no_grad():
    num_samples = 10  # Number of samples to generate
    latent_samples = torch.randn(num_samples, latent_dim)
    generated_images = vae_model.decoder(latent_samples)

plot_images(generated_images, title='Generated Samples')