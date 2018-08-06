import sys
import requests.packages.urllib3
import click
import logging
from keras.models import load_model
import matplotlib.pyplot as plt
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.classes import Collection, ToyData, deprocess

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    '--dataset', default='ToyData',
    help="Number of images to download"
)
@click.option(
    '--model', required=True,
    help="Model to inspect"
)
def main(dataset, model):
    logger.info('Initializing inspect script')
    dataset = eval(dataset)
    train_gen, val_gen, split = dataset.generators(128, 50)
    modelname = 'test.pkl'
    model = load_model(f'models/{modelname}')
    data = next(train_gen())[0][:5]
    decoded_data = model.predict(data)

    fig, ax = plt.subplots(
        2, 5, figsize=(10, 4)
    )
    for i in range(5):
        ax[0][i].imshow(
            deprocess(data[i]))

        ax[0][i].axis('off')
        ax[1][i].imshow(
            deprocess(decoded_data[i]))
        ax[1][i].axis('off')

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/inspect.log',
        level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s | "
            "%(levelname)s: %(message)s"
        )
    )
    main()
