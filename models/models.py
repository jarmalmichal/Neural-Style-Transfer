import torch
from torchvision.models import vgg19, vgg16


def load_model(model_name):
    if model_name == "vgg19":
        model = vgg19(weights="DEFAULT").features
    elif model_name == "vgg16":
        model = vgg16(weights="DEFAULT").features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    return model


def extract_style_features(model, image):
    # Mapping of layers used for style as in Gatys et al.
    mapping = {
        "0": "conv1_1",
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "28": "conv5_1",
    }
    layers = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in mapping:
            layers[mapping[name]] = x

    return layers


def extract_content_features(model, image):
    # Content representation layer as in Gatys et al.
    mapping = {"21": "conv4_2"}
    layers = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in mapping:
            layers[mapping[name]] = x

    return layers
