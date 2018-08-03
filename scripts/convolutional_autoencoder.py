import sys
import requests.packages.urllib3
import click
import os
import logging
from joblib import Parallel, delayed
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Collection, ToyData
import keras
from src.models import ConvolutionalAutoencoder

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--dataset', default='ToyData',
    help="Number of images to download"
)

def main(dataset):
    os.makedirs('data/images', exist_ok=True)
    logger.info('Initializing download script')
    patchset = eval(dataset).generate_patchset(128, 10)

    ca = ConvolutionalAutoencoder(dim=128, patchsize=128)
    ca.fit_generator(patchset)
    import pdb; pdb.set_trace()
    # for patch in patchset:
    #     import pdb; pdb.set_trace()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/train.conf', level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s"
            " | %(levelname)s: %(message)s"
        )
    )
    main()
