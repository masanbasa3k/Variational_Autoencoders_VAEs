import matplotlib.pyplot as plt
import numpy as np
import torch
from main import VAE, dataloader

input_dim = (3, 64, 64)  # Input image dimensions (channels, height, width)
hidden_dim = 256
latent_dim = 20
vae_model = VAE(input_dim, hidden_dim, latent_dim)

# Load the trained model (you should load from where you saved during training)
vae_model.load_state_dict(torch.load('vae_model.pth'))
vae_model.eval()

# Reconstruction and Sampling
with torch.no_grad():
    # Get a random batch
    images, _ = next(iter(dataloader))

    # Perform reconstruction by passing the data through the model
    reconstructed, _, _ = vae_model(images)

    # Generate new expressions by sampling from a random point
    latent_samples = torch.randn_like(reconstructed)
    generated_images = vae_model.decoder(latent_samples)

# Visualizations
def plot_images(images, title):
    images = images / 2 + 0.5  # Undo the normalization
    np_images = images.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Original images
plot_images(images, title='Original Images')

# Reconstructions
plot_images(reconstructed, title='Reconstructions')

# Samples
plot_images(generated_images, title='Generated Samples')
