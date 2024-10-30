import torch
import torchvision.transforms.v2 as v2
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = v2.Compose(
        [
            v2.Resize((1024, 1024)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img).unsqueeze(0)

    return img


def compute_tv_loss(img):
    # Computes total variation loss
    batch_size = img.size(0)
    h_x = img.size(2)
    w_x = img.size(3)
    diff_x = img[:, :, :, 1:] - img[:, :, :, :-1]
    diff_y = img[:, :, 1:, :] - img[:, :, :-1, :]
    tv_loss = (torch.sum(diff_x**2) + torch.sum(diff_y**2)) / (batch_size * h_x * w_x)

    return tv_loss


def gram_matrix(tensor):
    batch, channels, height, width = tensor.size()
    tensor = tensor.view(batch, channels, height * width)
    tensor_T = tensor.transpose(-2, -1)
    gram = torch.bmm(tensor, tensor_T)

    return gram


def convert_images(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def save_image(tensor, path):
    image = convert_images(tensor)
    image.save(path)
