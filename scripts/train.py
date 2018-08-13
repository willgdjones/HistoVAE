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
from src.classes import Dataset
from src.models import ConvolutionalAutoencoder

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--dataset_name', default='ToyData',
    help="Number of images to download"
)
@click.option(
    '--model_name', default='ConvolutionalAutoencoder',
    help="Number of images to download"
)
@click.option(
    '--inner_dim', default=512,
    help=(
        "Dimension of the inner vector"
        "Only relevant to deep learning models"
    )
)
@click.option(
    '--epochs', default=100,
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
    '--patch_size', default=128,
    help="Patchsize to use"
)
@click.option(
    '--lr', default=0.0002,
    help="Learning rate to use"
)
@click.option(
    '--beta_1', default=0.05,
    help="Beta 1 to use"
)
@click.option(
    '--batch_size', default=64,
    help="Beta 1 to use"
)
@click.option(
    '--dropout_rate', default=.0,
    help="Dropout rate to use"
)
def main(dataset_name, model_name, inner_dim, patch_size, epochs,
         n_patches, lr, beta_1, batch_size, dropout_rate):
    np.random.seed(42)
    os.makedirs('data/images', exist_ok=True)
    dataset = Dataset(n_images=10, n_tissues=6)
    Model = eval(model_name)
    logger.debug('Initializing download script')

    N = dataset.T * dataset.K * batch_size

    m = Model(inner_dim=inner_dim)
    data_filename = f'.cache/{dataset_name}_{patch_size}_{n_patches}.pkl'
    if isfile(data_filename):
        logger.debug(f'Loading data from cache')
        data = joblib.load(open(data_filename, 'rb'))
    else:

        data = dataset.sample_data(patch_size, int(n_patches))
        logger.debug(f'Saving data to cache')
        joblib.dump(data, open(data_filename, 'wb'))

    patches_data, imageIDs_data = data
    N = patches_data.shape[0]
    assert N == imageIDs_data.shape[0]
    p = np.random.permutation(N)
    patches_data, imageIDs_data = patches_data[p], imageIDs_data[p]

    params = {
        'lr': lr,
        'beta_1': beta_1,
        'epochs': epochs,
        'batch_size': batch_size,
        'patch_size': patch_size,
        'inner_dim': inner_dim,
        'dropout_rate': dropout_rate
    }

    m.train_on_data(
        patches_data, params
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
