import torch
from torch.optim import Adam, LBFGS
import os
import argparse

from src.models import models
from src.utils import utils
from src.train import style_transfer
from src.config import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img", type=str, help="content image name")
    parser.add_argument("--style_img", type=str, help="style image name")
    parser.add_argument(
        "--target_img",
        type=str,
        choices=["content", "style", "random"],
        default="content",
        help="target image initialization method",
    )
    parser.add_argument(
        "--model", type=str, choices=["vgg19", "vgg16", "alexnet"], default="vgg19"
    )
    parser.add_argument("--content_weight", type=float, default=1e5)
    parser.add_argument("--style_weight", type=float, default=1e5)
    parser.add_argument(
        "--tv_weight", type=float, default=1e-1, help="weight for total variation loss"
    )
    parser.add_argument("--optimizer", type=str, choices=["adam"], default="adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--steps", type=int, default=3000, help="number of iterations")

    args = parser.parse_args()

    # Check for cuda or mps for MacOS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    content_path = os.path.join(config.CONTENT_DIR, args.content_img)
    style_path = os.path.join(config.STYLE_DIR, args.style_img)

    content_img = utils.load_image(content_path).to(device)
    style_img = utils.load_image(style_path).to(device)

    if args.target_img == "content":
        target_img = content_img.clone()
    elif args.target_img == "style":
        target_img = style_img.clone()
    else:
        target_img = torch.randn_like(content_img)

    target_img.requires_grad_(True)

    if args.model in ["vgg19", "vgg16"]:
        layer_style_weights = config.VGG_STYLE_WEIGHTS
    else:
        layer_style_weights = config.ALEXNET_STYLE_WEIGHTS

    model = models.load_model(args.model).to(device)

    if args.optimizer == "adam":
        optimizer = Adam([target_img], lr=args.lr)
    else:
        pass

    print(f"Running on {device} device")
    stylized_img = style_transfer(
            content_img,
            style_img,
            target_img,
            model,
            args.model,
            optimizer,
            layer_style_weights,
            args.style_weight,
            args.content_weight,
            args.tv_weight,
            args.steps,
    )

    # Generate output filename based on input images
    content_name = os.path.splitext(args.content_img)[0]
    style_name = os.path.splitext(args.style_img)[0]
    output_filename = f"{content_name}_stylized_by_{style_name}_using_{args.model}.jpg"
    output_path = os.path.join(config.RESULTS_DIR, output_filename)

    # Save the output image in the results directory
    utils.save_image(stylized_img, output_path)
    print(f"Stylized image saved as {output_path}")
