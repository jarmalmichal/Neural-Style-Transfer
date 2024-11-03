import torch
from torchvision.models import vgg19, vgg16, alexnet


def load_model(model_name):
    if model_name == "vgg19":
        model = vgg19(weights="DEFAULT").features
    elif model_name == "vgg16":
        model = vgg16(weights="DEFAULT").features
    elif model_name == "alexnet":
        model = alexnet(weights="DEFAULT").features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    return model


def extract_vgg_features(model, image, mode="all"):
    """
    Extract feature maps based on mode
    mode: 'all', 'content', 'style'

    Layer choice (both style and content) are based on Gatys et al.
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    """
    if mode == "content":
        content_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            # Content representation layer same as in Gatys et al.
            if name == "21":
                content_features["conv4_2"] = x

        return content_features

    elif mode == "style":
        # Layers used for style same as in Gatys et al.
        style_layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "28": "conv5_1",
        }
        style_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in style_layers:
                style_features[style_layers[name]] = x

        return style_features

    else:
        # Need both for target image
        style_layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "28": "conv5_1",
        }
        content_features = {}
        style_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in style_layers:
                style_features[style_layers[name]] = x
            if name == "21":
                content_features["conv4_2"] = x

        return content_features, style_features


def extract_alexnet_features(model, image, mode="all"):
    """
    Extract feature maps based on mode
    mode: 'all', 'content', 'style'

    In case of AlexNet layer choice must be different
    Selected layers have empirically shown good results
    """
    if mode == "content":
        content_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name == "8":
                content_features["conv4"] = x

        return content_features

    elif mode == "style":
        style_layers = {
            "0": "conv1",
            "3": "conv2",
            "6": "conv3",
            "8": "conv4",
            "10": "conv5",
        }
        style_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in style_layers:
                style_features[style_layers[name]] = x

        return style_features

    else:
        # Need both for target image
        style_layers = {
            "0": "conv1",
            "3": "conv2",
            "6": "conv3",
            "8": "conv4",
            "10": "conv5",
        }
        content_features = {}
        style_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in style_layers:
                style_features[style_layers[name]] = x
            if name == "8":
                content_features["conv4"] = x

        return content_features, style_features
