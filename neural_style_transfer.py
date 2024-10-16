import torch
from torch.optim import Adam, LBFGS
import numpy as np
import os
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img", type=str, help="content image name")
    parser.add_argument("--style_img", type=str, help="style image name")
    parser.add_argument("--target_img", type=str, choices=['random', 'content', 'style'], help="target image initialization method")

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg19', 'mobilenet_v3_large', 'convnext_base'], default='vgg19')
