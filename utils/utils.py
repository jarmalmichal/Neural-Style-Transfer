import os
from PIL import Image
from datetime import datetime
import torch
import torchvision.transforms.v2 as v2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = Image.open(path).convert('RGB')
    
    transform = v2.Compose([
        v2.Resize((420, 420)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    img = transform(img).unsqueeze(0)
    return img


def gram_matrix(tensor):
    batch, dim, height, width = tensor.size()
    tensor = tensor.view(batch, dim, height * width)
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


def save_image(tensor, path):
    image = convert_images(tensor)
    image.save(path)
