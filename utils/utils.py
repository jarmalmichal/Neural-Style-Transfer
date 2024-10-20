import os
from PIL import Image
from datetime import datetime
import torch
import torchvision.transforms.v2 as v2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = Image.open(path)

    # Apply transformations
    transform = v2.Compose(
        [v2.Resize((512, 512)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    img = transform(img).unsqueeze(0)

    return img


def generate_gaussian(tensor):
    batch, channels, height, width = tensor.size()
    img = torch.randn(batch, channels, height, width)

    return img


def gram_matrix(tensor):
    batch, channels, height, width = tensor.size()
    tensor = tensor.view(batch, channels, height * width)
    tensor_T = tensor.transpose(-2, -1)
    gram = torch.bmm(tensor, tensor_T)

    return gram


def convert_images(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = (image * 255).astype(np.uint8)
    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image)

    return image


def display_images(*images, titles=None):
    fig, axs = plt.subplots(1, 4, figsize=(16, 6))

    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis("off")
        if titles and i < len(titles):
            axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()