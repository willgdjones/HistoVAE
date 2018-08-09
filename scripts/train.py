import sys
import requests.packages.urllib3
import click
import os
import logging
import random
import numpy as np
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import ToyData
from src.models import ConvolutionalAutoencoder

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--datasetname', default='ToyData',
    help="Number of images to download"
)
@click.option(
    '--modelname', default='ConvolutionalAutoencoder',
    help="Number of images to download"
)
@click.option(
    '--dim', default=128,
    help=(
        "Dimension of the inner vector"
        "Only relevant to deep learning models"
    )
)
@click.option(
    '--epochs', default=20,
    help=(
        "Number of epochs"
    )
)
@click.option(
    '--n_patches', default=100,
    help=(
        "Number of patches to sample from each image"
    )
)
@click.option(
    '--patchsize', default=128,
    help="Patchsize to use"
)
def main(datasetname, modelname, dim, patchsize, epochs, n_patches):
    np.random.seed(42)
    os.makedirs('data/images', exist_ok=True)
    dataset = eval(datasetname)
    Model = eval(modelname)
    logger.debug('Initializing download script')
    batch_size = 128

    N = dataset.T * dataset.K * batch_size

    m = Model(
        dim=dim, patchsize=patchsize
    )

    train_patches_data, train_imageIDs_data, val_patches_data,\
        val_imageIDs_data, split = dataset.training_data(
                                                patchsize, int(n_patches)
                                            )
    combined_train_data = zip(train_patches_data, train_imageIDs_data)
    combined_val_data = zip(val_patches_data, val_imageIDs_data)
    random.shuffle(combined_train_data)
    random.shuffle(combined_val_data)
    train_patches_data, train_imageIDs_data = zip(*combined_train_data)
    val_patches_data, val_imageIDs_data = zip(*combined_val_data)

    m.train_on_data(
        train_patches_data, val_patches_data, split, batch_size, epochs
    )


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/train.conf', level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s"
            " | %(levelname)s: %(message)s"
        )
    )
    main()
