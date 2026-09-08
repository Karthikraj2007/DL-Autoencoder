# DL- Convolutional Autoencoder for Image Denoising

### Name: KARTHIKRAJ C

### Register Number: 212224230117

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Images are often corrupted by random noise during acquisition, transmission, or sensor capture. The goal is to construct and train a Convolutional Autoencoder (CAE) using PyTorch that takes artificially corrupted, noisy grayscale images as input and reconstructs clean, high-fidelity target images by learning compressed latent spatial representations.

The experiment utilizes the MNIST benchmark handwritten digits dataset consisting of 70,000 grayscale images


## DESIGN STEPS
### STEP 1: 

Import the necessary modules (torch, torch.nn, torchvision, matplotlib). Check for GPU availability (cuda) to accelerate computation. Load the MNIST training and test datasets using torchvision.datasets, applying transforms.ToTensor() to scale pixel values to the range $[0, 1]$. Wrap the data into DataLoader objects with a batch size of 128 (with shuffle enabled for training).

### STEP 2: 

Define a noise injection function add_noise that adds zero-mean Gaussian random noise ($0.5 \times \mathcal{N}(0, 1)$) to the clean input tensors. Use torch.clamp to restrict values within $[0.0, 1.0]$, ensuring numerical stability and valid image pixel boundaries.

### STEP 3: 

Design a symmetric DenoisingAutoencoder class inheriting from nn.Module

### STEP 4: 

Instantiate the autoencoder model and move it to the target compute device (device). Define the loss function as Mean Squared Error (nn.MSELoss()) to measure pixel-wise reconstruction error between the reconstructed output and the original clean image. Initialize the optim.Adam optimizer with a learning rate of $0.001$.

### STEP 5: 

Train the network over 5 epochs. In each iteration, transfer clean images to the device, generate corresponding noisy inputs on the fly, pass noisy inputs through the autoencoder, compute the MSE loss against the uncorrupted originals, compute gradients via loss.backward(), and update network weights using optimizer.step(). Print the average loss for each epoch.

### STEP 6: 
Switch the model to evaluation mode (model.eval()). Disable gradient computation (torch.no_grad()) and extract a sample batch of test images. Pass the noisy test images through the trained model to obtain denoised reconstructions. Display a $3 \times 10$ Matplotlib subplot grid comparing the original clean images, noisy inputs, and reconstructed denoised outputs side by side.




## PROGRAM



```python

!pip install -q torchsummary

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
        )

        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


summary(model, (1, 28, 28))


def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()



    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

       
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)


```

### OUTPUT

### Model Summary
<img width="677" height="482" alt="image" src="https://github.com/user-attachments/assets/c2204b2b-2631-46a4-b439-416441b7d73d" />


### Training loss
<img width="342" height="117" alt="image" src="https://github.com/user-attachments/assets/75db7967-02a3-4586-b6bc-09e25e2fb98e" />


## Original vs Noisy Vs Reconstructed Image
<img width="1682" height="578" alt="image" src="https://github.com/user-attachments/assets/0d5a294b-7b6b-4911-81d6-54850c91457d" />

## RESULT
Thus, a Convolutional Denoising Autoencoder using PyTorch was successfully executed.
