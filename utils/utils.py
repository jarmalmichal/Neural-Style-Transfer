import re
import os
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
import torch
import torchvision.transforms.v2 as v2
import numpy as np


def load_image(path):
    # Check if the input is a URL
    url_pattern = re.compile(r"^https?://")

    if url_pattern.match(path):
        # Load image from URL
        response = requests.get(path)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
        else:
            raise ValueError(
                f"Failed to load image from URL. Status code: {response.status_code}"
            )
    else:
        img = Image.open(path)

    # Apply transformations
    transform = v2.Compose(
        [v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
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


def save_images(image):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = os.path.join(current_dir, "..", "..", "Images", "Results")

    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.png"

    full_path = os.path.join(folder_path, filename)

    image.save(full_path, format="PNG")
