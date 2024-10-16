import re
import requests
from PIL import Image
from io import BytesIO
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
            im = Image.open(BytesIO(response.content))
        else:
            raise ValueError(
                f"Failed to load image from URL. Status code: {response.status_code}"
            )
    else:
        im = Image.open(path)

    # Apply transformations
    transform = v2.Compose(
        [v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    img = transform(im).unsqueeze(0)

    return img


def gram_matrix(tensor):
    batch, channels, height, width = tensor.size()
    tensor = tensor.view(batch, channels, height * width)
    tensor_T = tensor.transpose(-2, -1)
    gram = torch.bmm(tensor, tensor_T)

    return tensor, tensor_T, gram


def convert_images(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = (image * 255).astype(np.uint8)
    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image)

    return image
