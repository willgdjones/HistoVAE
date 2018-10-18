
**This project has moved to https://gitlab.com/willgdjones/HistoVAE**

# HistoVAE
This repository outlines the HistoVAE project. Learning representations from histopathology images using unsupervised deep learning to create utility out of unlabelled histopathology images.

We use the annotated histopathology images from the GTEx project (Genotype-Tissue Expression Project).

## Table of Contents
1. [Download](#download)
2. [Patch Coordinates](#patchcoordinates)

## <a id='download'></a>Download
To download a small version of the dataset from NCI, run:

```
python scripts/download.py --n_images 50 --n_tissues 10
```

This will download 50 images from the 10 tissues with the greatest number of samples. Samples are sorted by `donorID` to ensure replicability.

## <a id='patchcoordinates'></a>Patch Coordinates
Much of the tissue image is whitespace. We segment the foreground and background of the tissue slice using Otsu thresholding. We sample square pixel patches of sizes 128, 256, 512 and 1024 pixels. We reject any samples where more than 25% of the patch is whitespace, defined as being above the in the Otsu background. We store the coordinates of the selected patches in HDF5 files within the `data/patches` directory using `pytables`. Using these coordinates, and knowing the patch size, we efficiently retrieve sampled patches at any level from the image using the [OpenSlide DeepZoomGenerator](#https://openslide.org/api/python/#module-openslide.deepzoom).

## Training the Convolutional Autoencoder
We train a Convolutional Autoencoder with convolutional layer in both the encoder and the decoder. After each convolutional layer in the encoder, we perform 2D max-pooling. After each convolutional layer in the decoder, we perform a 2D up-sampling. These operations in the decoder are equivalent to a deconvolutional layer. In the final layer of the encoder, and the first layer of the decoder, we perform dropout with probability 0.5. We use L2 regularization on the final encoded representation, and vary the dimension of this final representation to be a vector of length 256, 512 or 1024.

We augment patches passed through the autoencoder by performing horizontal and vertical flips of the patch each with probability 0.5. During training, we use the Adam optimizer with a learning rate of 0.0001, and a beta of 0.5. We found the performance of the model to be sensitive to these hyperparameters. For example, when using learning rate of 0.0005, we noticed stochastic jumps during training. We use a batch-size of 64. We used 128 filter for the first convolutional layer, 64 filters for the second, 32 for the third, and 16 for the last convolutional layer. The order of filters was reversed in the decoding layer. Receptive fields of size (3, 3) are using throughout.

## Viewing decoded encodings.

We generate realistic encodings on test images.

We perform Principal Component Analysis on the encoded representations.
