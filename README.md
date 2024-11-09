A PyTorch implementation of Neural Style Transfer
loosely based on the paper Image Style Transfer Using Convolutional Neural Networks by Gatys et al. but also includes TV (total variation) loss introduced in NST in Perceptual Losses for Real-Time Style Transfer and Super-Resolution by Johnson et al. This implementation supports multiple models (VGG19, VGG16 and AlexNet), optimization methods (LBFGS, Adam) and target image initialization method (using content or style images as target or gaussian noise).


**Key Features**

* Multiple models support (VGG19, VGG16, AlexNet)
* Different optimizers (LBFGS, Adam)
* Different target image initialization method
* Total Variation loss
* Configurable content, style and tv weights, learning rate (for Adam) and number of steps
* MPS support for MacOS (besides CUDA)


**Installation**

1. Clone the repository:

    git clone https://github.com/michaljarmal/Neural-Style-Transfer.git

2. Navigate to project directory:

    cd Neural-Style-Transfer

2. Create conda environment (all dependencies will be installed automatically from environment.yml file):

    conda env create -f environment.yml

3. Activate conda environment

    conda activate nst



**Images used:**

Content images:

* Photo by Sophie Otto: https://www.pexels.com/photo/city-hall-in-hamburg-by-the-river-20347993/

* Photo by Jeswin Thomas: https://www.pexels.com/photo/mosque-1007426/

* Photo by Mateusz Sałaciak: https://www.pexels.com/photo/white-and-brown-concrete-building-beside-body-of-water-4275890/

* Photo by Ilo Frey: https://www.pexels.com/photo/photo-of-yellow-and-blue-macaw-with-one-wing-open-perched-on-a-wooden-stick-2317904/

Style images:

* The Starry Night - Vincent van Gogh, Public domain, via Wikimedia Commons

* 