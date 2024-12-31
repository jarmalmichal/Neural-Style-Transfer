A PyTorch implementation of Neural Style Transfer based on Gatys et al. "Image Style Transfer Using Convolutional Neural Networks", with total variation loss inspired by Johnson et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution". This implementation includes multiple models, optimization and initialization methods.

![seamless_collage](https://github.com/user-attachments/assets/1dd1f45f-44e9-4aa3-b338-4cbd129a56a8)

**Key Features**

* Multiple models support (VGG19, VGG16, AlexNet)
* Different optimizers (LBFGS, Adam)
* Different target image initialization methods
* Total Variation loss
* Configurable weights for content, style, and TV loss
* Adjustable learning rate (for Adam optimizer) and number of steps


**Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/michaljarmal/Neural-Style-Transfer.git
    cd Neural-Style-Transfer
    ```

2. Create and activate conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate nst
    ```

**Usage**

1. Place your content and style images in the respective directories:
    * Content images in images/content/
    * Style images in images/style/

2. Run the style transfer (only image names required, not full paths):
    ```bash
    python3 neural_style_transfer.py --content_img your_content_img.jpg --style_img your_style_img.jpg
    ```


**Command Line Arguments**
| Argument | Description |
|----------|-------------|
| `--content_img` | Content image name |
| `--style_img` | Style image name |
| `--target_img` | Target image initialization method (`content`/`style`/`random`) |
| `--model` | Model choice (`vgg19`/`vgg16`/`alexnet`) |
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

* Adam: Requires more steps but it's faster and more memory efficient. In theory LBFGS being quasi-newton method can find more optimal solution but in practice results are nearly the same (see results dir, exactly the same hyperparams setting). It requires lr (and in case of some images lr needs to be tweaked differently) as with all first-order methods but in case of NST learning rate can be used as a "artistic tool" so to say. Different lr values will yield different results. In the end it all depends on subjective effect we want to achieve.


**Image Initialization**
* content: Initialize with content image (default, usually best results, see results dir)
* style: Initialize with style image (requires more steps)
* random: Initialize with random (gaussian) noise (requires more steps)


**Weight Tuning**
* Increase **content_weight** to preserve more content details
* Increase **style_weight** for stronger stylization
* Adjust **tv_weight** to control smoothness

**Acknowledgments**
* Implementation based on the paper by Gatys et al. Image Style Transfer Using Convolutional Neural Networks - https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
* Total Variation loss implementation inspired by Johnson et al. Perceptual Losses for Real-Time Style Transfer and Super-Resolution - https://arxiv.org/abs/1603.08155
 
These repos were useful:
* [pytorch-neural-style-transfer] (https://github.com/gordicaleksa/pytorch-neural-style-transfer)
* [style-transfer] (https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer)

Content images:

* Photo by Sophie Otto: https://www.pexels.com/photo/city-hall-in-hamburg-by-the-river-20347993/

* Photo by Ilo Frey: https://www.pexels.com/photo/photo-of-yellow-and-blue-macaw-with-one-wing-open-perched-on-a-wooden-stick-2317904/

* Photo by Pixabay: https://www.pexels.com/photo/church-beside-sea-532581/

* Neckarfront in Tübingen, a standard image in style transfer literature (Gatys et al., 2015; Gatys et al., 2016; Johnson et al., 2016)

Style images:

* The Starry Night, Vincent van Gogh, Public domain, via Wikimedia Commons

* The Great Wave off Kanagawa, After Katsushika Hokusai, Public domain, via Wikimedia Commons

* Adolphe Joseph Thomas, Bouquet of Flowers, WikiArt Dataset - https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

* Edvard Munch, 1893, The Scream, Public domain, via Wikimedia Commons

* Vassily Kandinsky, 1913, Composition 7, Public domain, via Wikimedia Commons

* Juan Gris, 1912, Portrait of Pablo Picasso, Public domain, via Wikimedia Commons
