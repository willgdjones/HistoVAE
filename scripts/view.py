import sys
import requests.packages.urllib3
import click
import logging
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
from os.path import isfile
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Dataset, deprocess

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--dataset_name', default='ToyData',
    help="Number of images to download"
)
@click.option(
    '--model_name', required=True,
    help="Model to inspect"
)
@click.option(
    '--patch_size', default=128,
    help="Patch size used"
)
@click.option(
    '--n_patches', default=10,
    help="Number of patches to sample from each image"
)
def main(dataset_name, model_name, patch_size, n_patches):
    logger.info('Initializing inspect script')
    dataset = Dataset(K=10, T=6)
    data_filename = f'.cache/{dataset_name}_{patch_size}_{n_patches}.pkl'
    if isfile(data_filename):
        logger.debug(f'Loading data from cache')
        data = joblib.load(open(data_filename, 'rb'))
    else:

        data = dataset.sample_data(patch_size, 10)
        logger.debug(f'Saving data to cache')
        joblib.dump(data, open(data_filename, 'wb'))
    patches_data, imageIDs_data = data
    model = load_model(f'models/{model_name}')
    patches = data[0][:5]
    decoded_patches = model.predict(patches)

    fig, ax = plt.subplots(
        2, 5, figsize=(10, 4)
    )
    for i in range(5):
        ax[0][i].imshow(
            deprocess(patches[i]))

        ax[0][i].axis('off')
        ax[1][i].imshow(
            deprocess(decoded_patches[i]))
        ax[1][i].axis('off')

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/inspect.log',
        level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s | "
            "%(levelname)s: %(message)s"
        )
    )
    main()
