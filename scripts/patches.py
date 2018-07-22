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
from src.classes import Collection, Coord
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    '--parallel', default=False,
    help="Run script in parallel where possible"
)
@click.option(
    '--n_jobs', default=4,
    help="Number of images to download"
)
def main(parallel, n_jobs):
    logger.info('Initializing patches script')
    lung_samples = Collection.where(
        'samples', lambda s: (
            s.tissue == 'Lung' and
            s.has_image() and
            s.has_expression()
        )
    )
    lung_images = [x.get_image() for x in lung_samples][:10]

    results = [image.download() for image in lung_images]
    assert all(results), "Some images failed to download"

    if parallel:
        logger.debug(f"Parallel execution with {n_jobs} jobs")
        image_patches = Parallel(
            n_jobs=n_jobs, backend='multiprocessing'
            )(delayed(
                lambda image: image.get_patches()
            )(image) for image in lung_images
        )
    else:
        logger.debug("Serial execution")
        image_patches = [
            p.get_patches() for p in lung_images
        ]
    import pdb; pdb.set_trace()





    import pdb; pdb.set_trace()


#
#
# plt.imshow(cv2.bitwise_and(region, region, mask=mask))
# plt.show()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/patcheslog.conf', level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(processName)s | %(levelname)s: %(message)s"
    )
    main()
