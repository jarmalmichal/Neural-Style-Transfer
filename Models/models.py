import torch
from torchvision.models import vgg19

def load_model(model_name):
    if model_name == "vgg19":
        model = vgg19(weights="DEFAULT").features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    return model


def extract_vgg_features(model, image):
    # Define names mapping in the manner of Gateys et al.
    mapping = {
        "0": "conv1_1",
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "21": "conv4_2",  # Content representation
        "28": "conv5_1",
    }

    layers = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in mapping:
            layers[mapping[name]] = x

    return layers
