import sys
import logging
from openslide.deepzoom import DeepZoomGenerator
import cv2
import mahotas
import click
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
sys.path.append('.')
from src.classes import (
    Annotation, Collection, Coord, ToyData
)
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--dataset', default='ToyData',
    help="Dataset to use"
)
def main(dataset):
    logger.info('Initializing train script')
    dataset = eval(dataset)
    patchset = dataset.generate_patchset()



    import pdb; pdb.set_trace()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/trainlog.conf',
        level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s |"
            "%(levelname)s: %(message)s"
        )
    )
    main()
