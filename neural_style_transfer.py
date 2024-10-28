import torch
from torch.optim import Adam
from torch.nn import MSELoss
import os
import argparse
from models import models
from utils import utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CONTENT_DIR = os.path.join(IMAGES_DIR, "content")
STYLE_DIR = os.path.join(IMAGES_DIR, "style")
RESULTS_DIR = os.path.join(IMAGES_DIR, "results")


def compute_tv_loss(img):
    # Calculate total variation loss
    batch_size = img.size(0)
    h_x = img.size(2)
    w_x = img.size(3)
    diff_x = img[:, :, :, 1:] - img[:, :, :, :-1]
    diff_y = img[:, :, 1:, :] - img[:, :, :-1, :]
    tv_loss = (torch.sum(diff_x**2) + torch.sum(diff_y**2)) / (batch_size * h_x * w_x)

    return tv_loss


def style_transfer(
    content_img,
    style_img,
    target_img,
    model,
    optimizer,
    layer_weights,
    style_weight,
    content_weight,
    tv_weight,
    steps,
):
    mse_loss = MSELoss()
    content_features = models.extract_content_features(model, content_img)
    style_features = models.extract_style_features(model, style_img)

    style_grams = {
        layer: utils.gram_matrix(style_features[layer]) for layer in style_features
    }

    for i in range(1, steps + 1):
        target_content = models.extract_content_features(model, target_img)
        target_styles = models.extract_style_features(model, target_img)

        content_loss = mse_loss(target_content["conv4_2"], content_features["conv4_2"])

        style_loss = 0
        for layer in layer_weights:
            target_style = target_styles[layer]
            _, dim, height, width = target_style.shape
            target_gram = utils.gram_matrix(target_style)
            style_gram = style_grams[layer]
            layer_style_loss = layer_weights[layer] * mse_loss(target_gram, style_gram)
            style_loss += layer_style_loss / (dim * height * width)

        tv_loss = compute_tv_loss(target_img)

        total_loss = (
            style_weight * style_loss
            + content_weight * content_loss
            + tv_weight * tv_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Iteration: {i}/{steps}")
            print(
                f"Total loss: {total_loss:.4f}, Content loss: {content_loss.item():.4f}, Style loss: {style_loss.item():.4f}, TV loss: {tv_loss.item():.4f}"
            )

    return target_img


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
        "--model", type=str, choices=["vgg19", "vgg16"], default="vgg19"
    )
    parser.add_argument("--content_weight", type=float, default=1e5)
    parser.add_argument("--style_weight", type=float, default=1e5)
    parser.add_argument(
        "--tv_weight", type=float, default=1e-1, help="weight for total variation loss"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--steps", type=int, default=3000, help="number of iterations")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_path = os.path.join(CONTENT_DIR, args.content_img)
    style_path = os.path.join(STYLE_DIR, args.style_img)

    content_img = utils.load_image(content_path).to(device)
    style_img = utils.load_image(style_path).to(device)

    if args.target_img == "content":
        target_img = content_img.clone()
    elif args.target_img == "style":
        target_img = style_img.clone()
    else:
        target_img = torch.randn_like(content_img)

    target_img.requires_grad_(True)

    layer_weights = {
        "conv1_1": 1,
        "conv2_1": 0.75,
        "conv3_1": 0.2,
        "conv4_1": 0.2,
        "conv5_1": 0.2,
    }

    model = models.load_model(args.model).to(device)

    optimizer = Adam([target_img], lr=args.lr)

    stylized_img = style_transfer(
        content_img,
        style_img,
        target_img,
        model,
        optimizer,
        layer_weights,
        args.style_weight,
        args.content_weight,
        args.tv_weight,
        args.steps,
    )

    # Generate output filename based on input images
    content_name = os.path.splitext(args.content_img)[0]
    style_name = os.path.splitext(args.style_img)[0]
    output_filename = f"{content_name}_stylized_by_{style_name}_using_{args.model}.jpg"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    # Save the output image in the RESULTS directory
    utils.save_image(stylized_img, output_path)
    print(f"Stylized image saved as {output_path}")
