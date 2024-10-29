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


def extract_features(model, image, mode='all'):
    """
    Extract feature maps based on mode
    mode: 'all', 'content', 'style'
    """
    if mode == 'content':
        content_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            # Content representation layer same as in Gatys et al.
            if name == "21":
                content_features["conv4_2"] = x

        return content_features
            
    elif mode == 'style':
        # Layers used for style same as in Gatys et al.
        style_layers = {
            "0": "conv1_1",
            "5": "conv2_1", 
            "10": "conv3_1",
            "19": "conv4_1",
            "28": "conv5_1"
        }
        style_features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in style_layers:
                style_features[style_layers[name]] = x

        return style_features
        
    else:  
        # mode == 'all'
        # Need both content and style for target image
        style_layers = {
            "0": "conv1_1",
            "5": "conv2_1", 
            "10": "conv3_1",
            "19": "conv4_1",
            "28": "conv5_1"
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
