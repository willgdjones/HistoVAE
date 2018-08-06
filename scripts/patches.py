import sys
import logging
import click
sys.path.append('.')

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    '--dataset', default='ToyData',
    help="Dataset to use"
)
def main(dataset):
    logger.info('Initializing patches script')
    eval(dataset).get_patch_coords()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/patches.log', level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s |"
            "%(levelname)s: %(message)s"
        )
    )
    main()
