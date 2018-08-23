import sys
import requests.packages.urllib3
import click
import os
import logging
import numpy as np
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Dataset


logger = logging.getLogger(__name__)


def extract_params(param_string):
    params = [p.split(':') for p in param_string.split('|')]
    params = dict((p[0], eval(p[1])) for p in params)
    return params


@click.command()
@click.option(
    '--n_tissues', default=6,
    help="Number of tissues with most numbers of samples"
)
@click.option(
    '--n_images', default=10,
    help="Number of images per tissue"
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
    '--model_name', default='ConvolutionalAutoencoder',
    help="Number of images to download"
)
@click.option(
    '--param_string', default=(
        'inner_dim:512|epochs:100|lr:0.0001|beta_1:0.05|'
        'batch_size:64|dropout_rate:.0'
    ),
    help=(
        "Specify the hyperparameters of the model."
    )
)
def main(n_tissues, n_images, n_patches, patch_size, model_name, param_string):
    np.random.seed(42)
    os.makedirs('data/images', exist_ok=True)
    dataset = Dataset(n_tissues=n_tissues, n_images=n_images)
    Model = eval(model_name)
    logger.debug('Initializing download script')

    params = extract_params(param_string)

    N = dataset.n_tissues * dataset.n_images * params['batch_size']

    m = Model(inner_dim=params['inner_dim'])

    data = dataset.sample_data(patch_size, int(n_patches))
    patches_data, imageIDs_data = data
    N = patches_data.shape[0]
    assert N == imageIDs_data.shape[0]
    p = np.random.permutation(N)
    patches_data, imageIDs_data = patches_data[p], imageIDs_data[p]

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
