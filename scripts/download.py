import sys
import requests.packages.urllib3
import click
import os
import logging
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Collection, ToyData


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--datasetname', default='ToyData',
    help="Number of images to download"
)
def main(datasetname):
    os.makedirs('data/images', exist_ok=True)
    logger.info('Initializing download script')
    dataset = eval(datasetname)
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
