import sys
import logging
import click
import os
sys.path.append('.')
from src.classes import Collection, ToyData

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--datasetname', default='ToyData',
    help="Dataset to use"
)
def main(datasetname):
    os.makedirs('data/patches', exist_ok=True)
    logger.info('Initializing patches script')
    dataset = eval(datasetname)
    dataset.get_patchcoordfiles()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/patches.log', level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s |"
            "%(levelname)s: %(message)s"
        )
    )
    main()
