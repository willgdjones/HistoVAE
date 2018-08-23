import sys
import os
import requests.packages.urllib3
import click
import logging
import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import isfile
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Dataset, deprocess

logger = logging.getLogger(__name__)
MODEL_PATH = 'models/'


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
    '--model_file', default=None,
    help="Model fie to use"
)
def main(n_tissues, n_images, n_patches, patch_size, model_file):
    logger.info('Initializing inspect script')
    dataset = Dataset(n_tissues=n_tissues, n_images=n_images)
    data = dataset.sample_data(patch_size, 15)
    patches_data, imageIDs_data = data
    K = 15
    N = patches_data.shape[0]
    idx = np.random.choice(range(N), K)
    patches = patches_data[idx]
    if model_file:
        fig, ax = plt.subplots(
            2, K, figsize=(1, 4)
        )
        model = load_model(MODEL_PATH + f'{model_file}.pkl')
        decoded_patches = model.predict(patches)
        fig.suptitle(model_file)
        for i in range(K):
            ax[0][i].imshow(deprocess(patches[i]))
            ax[0][i].axis('off')
            ax[1][i].imshow(deprocess(decoded_patches[i]))
            ax[1][i].axis('off')
        plt.savefig(f'figures/{model_file}.png')
    else:
        model_files = sorted(os.listdir(MODEL_PATH))
        n = len(model_files)
        fig, ax = plt.subplots(
            2 * n, K, figsize=(8, 4 * n)
        )
        for (k, model_file) in enumerate(model_files):
            model_name = model_file.replace('.pkl', '')
            model = load_model(MODEL_PATH + f'{model_name}.pkl')
            logger.debug(f'Generating decodings for {model_file}')
            decoded_patches = model.predict(patches)
            for i in range(K):
                ax[2*k][i].imshow(deprocess(patches[i]))
                ax[2*k][i].axis('off')
                if i == int(K/2):
                    ax[2*k][i].set_title(model_file)
                ax[2*k + 1][i].imshow(deprocess(decoded_patches[i]))
                ax[2*k + 1][i].axis('off')
        plt.savefig(f'figures/all_models.png')


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/view.log',
        level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s | "
            "%(levelname)s: %(message)s"
        )
    )
    main()
