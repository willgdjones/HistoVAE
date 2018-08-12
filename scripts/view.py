import sys
import requests.packages.urllib3
import click
import logging
from keras.models import load_model
import matplotlib.pyplot as plt
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Collection, ToyData, deprocess

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--dataset', default='ToyData',
    help="Number of images to download"
)
@click.option(
    '--modelname', required=True,
    help="Model to inspect"
)
@click.option(
    '--patch_size', default=128,
    help="Patch size used"
)
def main(dataset, modelname, patch_size):
    logger.info('Initializing inspect script')
    dataset = eval(dataset)
    data = dataset.sample_data(128, 50)
    patches_data, imageIDs_data = data
    model = load_model(f'models/{modelname}')
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
