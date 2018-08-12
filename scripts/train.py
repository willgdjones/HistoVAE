import sys
import requests.packages.urllib3
import click
import os
from os.path import isfile
import logging
import random
import numpy as np
import joblib
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
    batch_size = 64

    N = dataset.T * dataset.K * batch_size

    m = Model(
        dim=dim, patchsize=patchsize
    )
    data_filename = f'.cache/{datasetname}_{patchsize}_{n_patches}.pkl'
    if isfile(data_filename):
        logger.debug(f'Loading data from cache')
        data = joblib.load(open(data_filename, 'rb'))
    else:

        data = dataset.sample_data(patchsize, int(n_patches))
        logger.debug(f'Saving data to cache')
        joblib.dump(data, open(data_filename, 'wb'))

    patches_data, imageIDs_data = data
    N = patches_data.shape[0]
    assert N == imageIDs_data.shape[0]
    p = np.random.permutation(N)
    patches_data, imageIDs_data = patches_data[p], imageIDs_data[p]

    m.train_on_data(
        patches_data, batch_size, epochs
    )

    m.save()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/train.conf', level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s"
            " | %(levelname)s: %(message)s"
        )
    )
    main()
