import sys
import logging
from openslide.deepzoom import DeepZoomGenerator
import cv2
import mahotas
import click
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
sys.path.append('.')
from src.classes import Dataset, deprocess

from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.command()
def main(dataset):
    logger.info('Initializing debug script')
    dataset = Dataset(n_tissues=6, n_images=10)
    data = dataset.sample_data(256, 50)
    patches_data, imageIDs_data = data
    import pdb; pdb.set_trace()
    # np.save('images.npy', 255 - patches_data[:100])
    # for i in range(10):
    #     scipy.misc.imsave(f'tmp/test{i}.png', 255 - patches_data[i])


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/debug.log',
        level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s |"
            "%(levelname)s: %(message)s"
        )
    )
    main()
