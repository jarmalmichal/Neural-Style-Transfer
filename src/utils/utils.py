import torch
import torchvision.transforms.v2 as v2
import numpy as np
from PIL import Image
from src.config import config


def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = v2.Compose(
        [
            v2.Resize((1024, 1024)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )
    img = transform(img).unsqueeze(0)

    return img


def compute_tv_loss(img):
    """
    Calculate total variation loss

    Based on Johnson et al.
    In this implementation it's used as a loss and not as a regularizer but effect is the same

    https://arxiv.org/abs/1603.08155
    """
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
    image = tensor.to("cpu").clone().detach().numpy().squeeze()
    # Transpose because numpy expects H, W, C
    image = image.transpose(1, 2, 0)
    # Unnormalize
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def save_image(tensor, path):
    image = convert_images(tensor)
    image.save(path)
