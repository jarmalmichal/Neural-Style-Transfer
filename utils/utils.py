import torch
from torchvision.transforms import v2
from PIL import Image


def load_image(path):
    im = Image.open(path)
    transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    img = transform(im).unsqueeze(0)

    return img


def gram_matrix(tensor):
    batch, channels, height, width = tensor.size()
    tensor = tensor.view(batch, channels, height * width)
    tensor_T = tensor.transpose(-2, -1)
    gram = torch.bmm(tensor, tensor_T)

    return tensor, tensor_T, gram
