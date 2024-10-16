import torch
from torchvision.models import vgg19
from typing import Dict, Tuple, Union


def load_model(model_name: str) -> torch.nn.Module:
    if model_name == "vgg19":
        model = vgg19(weights="DEFAULT").features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model


model = load_model("vgg19")
print(model)


def extract_vgg_features(
    model: torch.nn.Module, image: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    # Define names mapping in the manner of Gateys et al.
    mapping = {
        "0": "conv1_1",
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "21": "conv4_2",
        "28": "conv5_1",
    }

    style_layers = {}
    content_layer = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in mapping:
            if name == "21":
                content_layer[mapping[name]] = x
            else:
                style_layers[mapping[name]] = x

    if not content_layer:
        raise ValueError("Content layer not found in the model")
    if not style_layers:
        raise ValueError("Style layers not found in the model")

    return style_layers, content_layer
