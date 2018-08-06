import sys
import requests.packages.urllib3
import click
import os
import logging
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import ToyData
from src.models import ConvolutionalAutoencoder

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--datasetname', default='ToyData',
    help="Number of images to download"
)
@click.option(
    '--modelname', default='ConvolutionalAutoencoder',
    help="Number of images to download"
)
@click.option(
    '--dim', default=128,
    help=(
        "Dimension of the inner vector"
        "Only relevant to deep learning models"
    )
)
@click.option(
    '--epochs', default=20,
    help=(
        "Number of epochs"
    )
)
@click.option(
    '--patchsize', default=128,
    help="Patchsize to use"
)
def main(datasetname, modelname, dim, patchsize, epochs):
    os.makedirs('data/images', exist_ok=True)
    dataset = eval(datasetname)
    Model = eval(modelname)
    logger.info('Initializing download script')
    batchsize = 50
    train_gen, val_gen, split =\
        dataset.generators(patchsize, batchsize)
    n = dataset.T * dataset.k * batchsize

    m = Model(
        dim=dim, patchsize=patchsize
    )
    m.train(
        train_gen, val_gen, split,
        n, batchsize, epochs
    )

if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/train.conf', level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s"
            " | %(levelname)s: %(message)s"
        )
    )
    main()
