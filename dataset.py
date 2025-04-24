import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

print("Length of Training Data:", len(training_data))
print("Length of Training Data:", len(test_data))

# actually opening the image
image, label = training_data[0]
image = image.squeeze().numpy()

# blurring using gaussian filter
blurred_image = gaussian_filter(image, sigma=2.0)
blurred_image_tensor = torch.tensor(blurred_image).unsqueeze(0)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# original image
axes[0].imshow(image, cmap='gray')
axes[0].set_title(f'Original Not Blurred: {label}')
axes[0].axis('off')

# blurred image
axes[1].imshow(blurred_image, cmap='gray')
axes[1].set_title(f'Blurred: {label}')
axes[1].axis('off')

plt.show()
