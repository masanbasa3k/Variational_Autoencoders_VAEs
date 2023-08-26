import os
import torch
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

# Visualize
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{class_names[labels[j]]:5s}' for j in range(batch_size)))


# How to Continue Loading Data for Model Training
# Here, we haven't trained a VAE model, but these steps would be used in the training process.
# You can feed the DataLoader into your model to start the training.