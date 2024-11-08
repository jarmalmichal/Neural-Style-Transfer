import torch
from torch.nn import MSELoss
from src.models import models
from src.utils import utils
from src.config import config


def compute_grams(style_features):
    # Precomputes gram matrices for style image
    style_grams = {
        layer: utils.gram_matrix(style_features[layer]) for layer in style_features
    }
    return style_grams


def extract_features(model, image, model_name, mode="all"):
    if model_name in ["vgg16", "vgg19"]:
        return models.extract_vgg_features(model, image, mode)
    else:
        return models.extract_alexnet_features(model, image, mode)


def compute_content_loss(mse_loss, target_features, content_features, model_name):
    if model_name in ["vgg16", "vgg19"]:
        return mse_loss(target_features["conv4_2"], content_features["conv4_2"])
    else:
        return mse_loss(target_features["conv4"], content_features["conv4"])


def style_transfer(
    content_img,
    style_img,
    target_img,
    model,
    model_name,
    optimizer,
    layer_style_weights,
    style_weight,
    content_weight,
    tv_weight,
    steps,
):
    mse_loss = MSELoss()

    # Extract feature maps that represent content and style for both images
    content_features = extract_features(model, content_img, model_name, mode="content")
    style_features = extract_features(model, style_img, model_name, mode="style")

    # Calculate grams
    style_grams = compute_grams(style_features)
    
    # Print frequency based on optimizer type
    print_freq = 100 if isinstance(optimizer, torch.optim.LBFGS) else 500

    if isinstance(optimizer, torch.optim.LBFGS):
        # Ensure target_img is contiguous for LBFGS
        target_img.data = target_img.data.contiguous()
        
        def closure():
            optimizer.zero_grad()
            
            # Extract feature maps that represent content and style for target image
            target_content_features, target_style_features = extract_features(
                model, target_img, model_name, mode="all"
            )

            # Calculate content loss as MSE
            content_loss = compute_content_loss(
                mse_loss, target_content_features, content_features, model_name
            )

            style_loss = 0
            for layer in layer_style_weights:
                target_style = target_style_features[layer]
                _, dim, height, width = target_style.shape

                # Calculate gram matrix for target image and extract precomputed gram for style image
                target_gram = utils.gram_matrix(target_style)
                style_gram = style_grams[layer]

                # Calculate weighted MSE loss between Gram matrices
                layer_style_loss = layer_style_weights[layer] * mse_loss(
                    target_gram, style_gram
                )

                # Normalize by feature map dimensions and add to "total" style loss
                style_loss += layer_style_loss / (dim * height * width)

            # Calculate total variation loss
            tv_loss = utils.compute_tv_loss(target_img)

            # Weight and combine all losses
            total_loss = (
                style_weight * style_loss
                + content_weight * content_loss
                + tv_weight * tv_loss
            )

            # Make sure gradients are contiguous
            total_loss.backward()
            if target_img.grad is not None:
                target_img.grad.data = target_img.grad.data.contiguous()
            
            closure.loss_values = {
                'total': total_loss.item(),
                'content': content_loss.item(),
                'style': style_loss.item(),
                'tv': tv_loss.item()
            }
            
            return total_loss

        for i in range(1, steps + 1):
            try:
                optimizer.step(closure)
            except RuntimeError as e:
                if "view size is not compatible" in str(e):
                    # If we still get view error, try to recover
                    target_img.data = target_img.data.contiguous()
                    if target_img.grad is not None:
                        target_img.grad.data = target_img.grad.data.contiguous()
                    optimizer.step(closure)
                else:
                    raise e
            
            if i % print_freq == 0:
                losses = closure.loss_values
                print(f"Iteration: {i}/{steps}")
                print(
                    f"Total loss: {losses['total']:.4f}, Content loss: {losses['content']:.4f}, "
                    f"Style loss: {losses['style']:.4f}, TV loss: {losses['tv']:.4f}"
                )
    else:
        for i in range(1, steps + 1):
            # Extract feature maps that represent content and style for target image
            target_content_features, target_style_features = extract_features(
                model, target_img, model_name, mode="all"
            )

            # Calculate content loss as MSE
            content_loss = compute_content_loss(
                mse_loss, target_content_features, content_features, model_name
            )

            style_loss = 0
            for layer in layer_style_weights:
                target_style = target_style_features[layer]
                _, dim, height, width = target_style.shape

                # Calculate gram matrix for target image and extract precomputed gram for style image
                target_gram = utils.gram_matrix(target_style)
                style_gram = style_grams[layer]

                # Calculate weighted MSE loss between Gram matrices
                layer_style_loss = layer_style_weights[layer] * mse_loss(
                    target_gram, style_gram
                )

                # Normalize by feature map dimensions and add to "total" style loss
                style_loss += layer_style_loss / (dim * height * width)

            # Calculate total variation loss
            tv_loss = utils.compute_tv_loss(target_img)

            # Weight and combine all losses
            total_loss = (
                style_weight * style_loss
                + content_weight * content_loss
                + tv_weight * tv_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                print(f"Iteration: {i}/{steps}")
                print(
                    f"Total loss: {total_loss:.4f}, Content loss: {content_loss.item():.4f}, "
                    f"Style loss: {style_loss.item():.4f}, TV loss: {tv_loss.item():.4f}"
                )

    return target_img