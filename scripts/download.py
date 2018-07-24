import sys
import requests.packages.urllib3
import click
import logging
from joblib import Parallel, delayed
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Collection, ToyData


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--dataset', default='ToyData',
    help="Number of images to download"
)
def main(dataset):
    logger.info('Initializing download script')
    dataset = eval(dataset)
    eval(dataset).download()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/downloadlog.conf', level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(processName)s | %(levelname)s: %(message)s"
    )
    main()
