
# HistoVAE
This repository outlines the HistoVAE project. Learning representations from histopathology images using unsupervised deep learning to enable to utilisation on unlabelled histopathology images.

We use the annotated histopathology images from the GTEx project (Genotype-Tissue Expression Project).

## Table of Contents
1. [Download](#download)
2. [Patch Coordinates](#patchcoordinates)

## <a id='patchcoordinates'></a>Download
To download the ToyData from NCI, run:
`python scripts/download.py --dataset ToyData`

ToyData consists of 10 samples from each of the 6 most common tissues. These tissues along with their frequencies are:

Each sample in the ToyData has annotated genetic and expression data.

## <a id='patchcoordinates'></a>Patch Coordinates
Much of the tissue image is whitespace. We segment the foreground and background of the tissue slice using an Otsu threshold. We sample square pixel patches of sizes 128, 256, 512 and 2014 pixels, and ensure that no more than 25% of the patches is whitespace. We store the coordinates of the selected patches in `data/patches`. Using these coordinates, and knowing the patch size, we can efficiently retrieve patches at any level from the image using the [OpenSlide DeepZoomGenerator](#https://openslide.org/api/python/#module-openslide.deepzoom).

## Training an Autoencoder
We train a Convolutional Autoencoder 
