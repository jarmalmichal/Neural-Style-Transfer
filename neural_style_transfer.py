import torch
from torch.optim import Adam, LBFGS
from torch.nn import MSELoss
import numpy as np
import os
import argparse
from Models import models
from utils import utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
CONTENT_DIR = os.path.join(IMAGES_DIR, "Content")
STYLE_DIR = os.path.join(IMAGES_DIR, "Style")


def optimize(content_img, style_img, target_img, model, optimizer, layer_style_weights, style_weight, content_weight, steps):
    for i in range(steps):
        content_img_layers = models.extract_vgg_features(model, content_img)
        style_img_layers = models.extract_vgg_features(model, style_img)
        target_img_layers = models.extract_vgg_features(model, target_img)

        content_loss = MSELoss(content_img_layers["conv4_2"], target_img_layers["conv4_2"])
        style_loss = 0.0
        for name in layer_style_weights:
            _, channels, height, width = style_img_layers[name].size()
            style_img_gram = utils.gram_matrix(style_img_layers[name])
            target_img_gram = utils.gram_matrix(target_img_layers[name])
            layer_style_loss = layer_style_weights[name] * MSELoss(style_img_gram, target_img_gram)
            style_loss += layer_style_loss / (channels * height * width)
        
        total_loss = style_weight * style_loss + content_weight * content_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch: [{i+1}/{steps}]")
        print(f"Total loss: {total_loss}")
    
    return target_img

     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img", type=str, help="content image name")
    parser.add_argument("--style_img", type=str, help="style image name")
    parser.add_argument(
        "--target_img",
        type=str,
        choices=["random", "content", "style"],
        help="target image initialization method",
        default="content",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vgg19"],
        default="vgg19",
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["lbfgs", "adam"], default="adam"
    )

    parser.add_argument("--content_weight", type=float, default=1)
    parser.add_argument("--style_weight", type=float, default=1e6)
    parser.add_argument("--steps", type=int, default=7000)

    args = parser.parse_args()

    content_path = os.path.join(CONTENT_DIR, args.content_img)
    style_path = os.path.join(STYLE_DIR, args.style_img)

    content_img = utils.load_image(content_path)
    style_img = utils.load_image(style_path)

    if args.target_img == "content":
        target_img = content_img
    elif args.target_img == "style":
        target_img = style_img
    else:
        target_img = utils.generate_gaussian(content_img)

    layer_style_weights = {
        "conv1_1": 1.0,
        "conv2_1": 0.75,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    }

    content_weight = args.content_weight
    style_weight = args.style_weight
    steps = args.steps
    model = models.load_model(args.model)
    if args.optimizer.lower() == "adam":
        optimizer = Adam([target_img])
    else:
        optimizer = LBFGS([target_img])
    

    stylized_img = optimize(content_img, style_img, target_img, model, optimizer, layer_style_weights, style_weight, content_weight, steps)