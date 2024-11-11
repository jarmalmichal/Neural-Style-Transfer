A PyTorch implementation of Neural Style Transfer
based on the paper Image Style Transfer Using Convolutional Neural Networks by Gatys et al. but with a twist. It also includes TV (total variation) loss inspired by Perceptual Losses for Real-Time Style Transfer and Super-Resolution by Johnson et al. This implementation also supports multiple models (VGG19, VGG16 and AlexNet), optimization methods (LBFGS, Adam) and target image initialization methods (using content or style images as target or gaussian noise).

![images](https://github.com/user-attachments/assets/5d93d330-b06f-48fc-a458-4c1f4a5b8b08)

**Key Features**

* Multiple models support (VGG19, VGG16, AlexNet)
* Different optimizers (LBFGS, Adam)
* Different target image initialization method
* Total Variation loss
* Configurable content, style and tv weights, learning rate (for Adam) and number of steps


**Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/michaljarmal/Neural-Style-Transfer.git
    cd Neural-Style-Transfer
    ```

2. Create and activate conda environment (all dependencies will be installed automatically from environment.yml file):
    ```bash
    conda env create -f environment.yml
    conda activate nst
    ```

**Usage**

1. Place your content and style images in the respective directories:
    * Content images in images/content/
    * Style images in images/style/

2. Run the style transfer:
    You only need to provide name of images. There is no need for full path
    ```bash
    python3 neural_style_transfer.py --content_img your_content_img.jpg --style_img your_style_img.jpg
    ```


**Comand Line Arguments**
| Argument | Description |
|----------|-------------|
| `--content_img` | Content image name |
| `--style_img` | Style image name |
| `--target_img` | Target image initialization method (`content`/`style`/`random`) |
| `--model` | Model to use (`vgg19`/`vgg16`/`alexnet`) |
| `--content_weight` | Weight for content loss |
| `--style_weight` | Weight for style loss |
| `--tv_weight` | Weight for total variation loss |
| `--optimizer` | Optimizer choice (`adam`/`lbfgs`) |
| `--lr` | Learning rate (for Adam) |
| `--steps` | Number of optimization steps |

**Advanced Configuration**

**Model Selection**
Implementation supports three models:
* VGG19 (default): Best results
* VGG16: Results are nearly identical (examples in results dir, exactly the same hyperparams setting) but it's slightly faster
* AlexNet: Much smaller model therefore it will produce lower quality results but they still look satisfying (examples in results dir). It requires more hyperparams tuning but it's much faster and computationally efficient.

**Optimizer Choice**
* LBFGS (default): Standard choice for NST. Requires less steps but uses more memory therefore each step takes longer and overall process takes more time. May yield slightly better results and it does not require lr as Adam.

* Adam: Requires more steps but it's faster and more memory efficient. In theory LBFGS being quasi-newton method can find more optimal solution but in practice results are nearly the same (see results dir, exactly the same hyperparams setting). It requires lr (and in case of some images lr needs to be tweaked differently) as with all first-order methods but in case of NST learning rate can be used as a "artistic tool" so to say. Different lr values will yield different results. In the end it all depends or subjective effect we want to achieve.


**Image Initialization**
* content: Initialize with content image (default, usually best results)
* style: Initialize with style image (requires more steps to achieve similar results to content init)
* random: Initialize with random (gaussian) noise (requires more steps to achieve similar results to content init)


**Weight Tuning**
* Increase **content_weight** to preserve more content details
* Increase **style_weight** for stronger stylization
* Adjust **tv_weight** to control smoothness

**Acknowledgments**
* Implementation based on the paper by Gatys et al. Image Style Transfer Using Convolutional Neural Networks - https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
* Total Variation loss implementation inspired by Johnson et al. Perceptual Losses for Real-Time Style Transfer and Super-Resolution - https://arxiv.org/abs/1603.08155

Content images:

* Photo by Sophie Otto: https://www.pexels.com/photo/city-hall-in-hamburg-by-the-river-20347993/

* Photo by Ilo Frey: https://www.pexels.com/photo/photo-of-yellow-and-blue-macaw-with-one-wing-open-perched-on-a-wooden-stick-2317904/

* Photo by Pixabay: https://www.pexels.com/photo/church-beside-sea-532581/

Style images:

* The Starry Night, Vincent van Gogh, Public domain, via Wikimedia Commons

* The Great Wave off Kanagawa, After Katsushika Hokusai, Public domain, via Wikimedia Commons

* Adolphe Joseph Thomas, Bouquet of Flowers, WikiArt Dataset - https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

* Edvard Munch, 1893, The Scream, Public domain, via Wikimedia Commons
