import sys
import requests.packages.urllib3
import click
import os
import logging
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Dataset


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--n_images', default=10,
    help="Number of images per tissue"
)
@click.option(
    '--n_tissues', default=6,
    help="Number of tissues with most numbers of samples"
)
def main(n_images, n_tissues):
    os.makedirs('data/images', exist_ok=True)
    logger.info('Initializing download script')
    dataset = Dataset(n_images=n_images, n_tissues=n_tissues)
    dataset.download()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/download.log',
        level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s | "
            "%(levelname)s: %(message)s"
        )
    )
    main()
