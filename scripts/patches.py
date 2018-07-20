import sys
import logging
from openslide.deepzoom import DeepZoomGenerator
import cv2
import mahotas
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
sys.path.append('.')
from src.classes import Collection, Coord

logger = logging.getLogger(__name__)

def main():
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

    patches = lung_images[0].get_patches()



    # tiles = [
    #     tile_generator.get_tile(level_count, (coord.x, coord.y))
    #     for coord in coords
    # ]
    # mask_coords = np.argwhere(mask > 0)


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
