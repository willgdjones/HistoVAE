import sys
import requests.packages.urllib3
import click
import logging
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Sample, Image, Donor, Annotations, Collection
from joblib import Parallel, delayed


logger = logging.getLogger(__name__)


def download_image(image):
    return image.download()


@click.command()
@click.option(
    '--n_image', default=10,
    help="Number of images to download"
)
@click.option(
    '--parallel', default=False,
    help="Run script in parallel where possible"
)
@click.option(
    '--n_jobs', default=4,
    help="Number of images to download"
)
def main(n_image, parallel, n_jobs):
    logger.info('Initializing download script')
    lung_samples = Collection.where(
        'samples', lambda s: (
            s.tissue == 'Lung' and
            s.has_image() and
            s.has_expression()
            )
    )
    lung_images = [x.get_image() for x in lung_samples][:n_image]

    logger.info(f"Starting download with {n_jobs} workers")
    if parallel:
        results = Parallel(
            n_jobs=n_jobs, backend='multiprocessing')(
            delayed(download_image)(image) for image in lung_images
        )
    else:
        results = [download_image(image) for image in lung_images]

    assert all(results), "Some images failed to download"


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/downloadlog.conf', level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(processName)s | %(levelname)s: %(message)s"
    )
    main()
