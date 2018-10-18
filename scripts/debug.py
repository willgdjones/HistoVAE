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
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


@click.command()
def main():
    logger.info('Initializing debug script')
    dataset = Dataset(n_tissues=6, n_images=10)
    data = dataset.sample_data(128, 50)
    patches_data, imageIDs_data = data
    for i in tqdm(range(len(imageIDs_data))):
        GTEx_ID = imageIDs_data[i]
        idx = i % 50
        scipy.misc.imsave(f'data/cellprofiler/patches/{i:04d}_{GTEx_ID}_{idx}.png', 255 - patches_data[i])

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
