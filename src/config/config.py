import os

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CONTENT_DIR = os.path.join(IMAGES_DIR, "content")
STYLE_DIR = os.path.join(IMAGES_DIR, "style")
RESULTS_DIR = os.path.join(IMAGES_DIR, "results")

# ImageNet specific mean and std
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model specific style weights
VGG_STYLE_WEIGHTS = {
    "conv1_1": 1,
    "conv2_1": 0.75,
    "conv3_1": 0.2,
    "conv4_1": 0.2,
    "conv5_1": 0.2,
}

ALEXNET_STYLE_WEIGHTS = {
    "conv1": 0.15,
    "conv2": 0.6,
    "conv3": 0.8,     
    "conv4": 0.4,
    "conv5": 0.3,     
}