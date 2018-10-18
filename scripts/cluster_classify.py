import sys
import logging
import hashlib
from openslide.deepzoom import DeepZoomGenerator
import cv2
import mahotas
import click
import pickle
import scipy.misc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from os.path import isfile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from collections import namedtuple
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.models import Model, load_model
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('.')
from src.classes import Dataset, deprocess, Image
from src.models import *


logger = logging.getLogger(__name__)
CACHE_PATH = '.cache/'


def get_tissue_IDs(dataset_name, image_objs, cache=True):

    caching_hash = hashlib.sha256(
        ''.join([x.ID.GTEx_ID for x in image_objs]).encode('utf-8')
    ).hexdigest()
    filepath = CACHE_PATH + dataset_name + f'_{caching_hash}_tissue_IDs.pkl'

    if isfile(filepath) and cache:
        logger.debug('Loading tissue IDs from cache')
        tissue_IDs = pickle.load(open(filepath, 'rb'))
    else:
        logger.debug('Generating tissue IDs')
        tissue_IDs = np.array([x.get_sample().tissue for x in image_objs])
        pickle.dump(tissue_IDs, open(filepath, 'wb'))

    return tissue_IDs


def generate_features(features_ID, patches, model_file_id):
    filepath = CACHE_PATH + features_ID + '.pkl'
    if isfile(filepath):
        logger.debug('Loading features from cache')
        features = pickle.load(open(filepath, 'rb'))
    else:
        logger.debug(f'Generating {model_file_id} features')
        m = load_model(
            'models/' + model_file_id + '.pkl'
        )
        encoder = Model(m.layers[0].input, m.layers[14].output)
        features = encoder.predict(patches)
        pickle.dump(features, open(filepath, 'wb'))
    return features


def plot_PCA(dataset_name, features, image_objs, mode):

    # pca = PCA()
    # pca_features = pca.fit_transform(features)
    #
    # fig1 = plt.figure(figsize=(15, 5))
    # fig1_ax1 = fig1.add_subplot(131, projection='3d')
    # fig1_ax1.scatter(
    #     pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
    #     c=GTEx_factor_IDs, s=5, alpha=0.3
    # )
    # fig1_ax2 = fig1.add_subplot(132, projection='3d')
    # fig1_ax2.scatter(
    #     pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
    #     c=tissue_factor_IDs, s=5, alpha=0.3
    # )
    # fig1_ax3 = fig1.add_subplot(133)
    #
    # fig1_ax3.bar(
    #     range(len(pca.explained_variance_)),
    #     np.cumsum(pca.explained_variance_)
    # )
    # plt.tight_layout()
    # plt.show()


    # fig2 = plt.figure(figsize=(10, 10))
    #
    # pca = PCA()
    # pca_features = pca.fit_transform(
    #     aggregated_features['GTEx_factor_IDs']['np.mean']
    # )
    #
    # fig2_ax1 = fig2.add_subplot(221, projection='3d')
    # fig2_ax1.scatter(
    #     pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
    #     c=tissue_factor_IDs, s=5, alpha=0.3
    # )
    #
    # pca = PCA()
    # pca_features = pca.fit_transform(
    #     aggregated_features['GTEx_factor_IDs']['np.median']
    # )
    #
    # fig2_ax2 = fig2.add_subplot(222, projection='3d')
    # fig2_ax2.scatter(
    #     pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
    #     c=np.unique(tissue_factor_IDs), s=5, alpha=0.3
    # )
    #
    # pca = PCA()
    # pca_features = pca.fit_transform(
    #     aggregated_features['tissue_factor_IDs']['np.mean']
    # )
    #
    # fig2_ax3 = fig2.add_subplot(223, projection='3d')
    # fig2_ax3.scatter(
    #     pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
    #     c=np.unique(tissue_factor_IDs), s=5, alpha=0.3
    # )
    #
    # pca = PCA()
    # pca_features = pca.fit_transform(
    #     aggregated_features['tissue_factor_IDs']['np.median']
    # )
    #
    # fig2_ax4 = fig2.add_subplot(224, projection='3d')
    # fig2_ax4.scatter(
    #     pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
    #     c=np.unique(tissue_factor_IDs), s=5, alpha=0.3
    # )



    factor_IDs, unique_factor_IDs = generate_factors(
        dataset_name, image_objs, mode
    )

    hot = plt.get_cmap('plasma')
    cNorm = colors.Normalize(vmin=0, vmax=len(unique_factor_IDs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
    # colors = ['blue', 'green', 'yellow', 'red', 'black', 'purple']
    pca = PCA()
    pca_features = pca.fit_transform(features)
    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # for f in np.unique(factor_IDs):
    #     color = colors[f]
    #     print(color)
    #     ax.plot(
    #         pca_features[:, 0], pca_features[:, 1], pca_features[:, 2],
    #         'o', color=color, label=colors[f]
    #     )
    ax = fig.add_subplot(111, projection='3d')
    for f in np.unique(factor_IDs):
        idx = factor_IDs == f
        ax.scatter(
            pca_features[idx, 0], pca_features[idx, 1], pca_features[idx, 2],
            color=scalarMap.to_rgba(f), label=unique_factor_IDs[f], alpha=0.5
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Hello')
    ax.legend()
    plt.show()



def get_variable_name(*variable):
    '''gets string of variable name
    inputs
        variable (str)
    returns
        string
    '''
    if len(variable) != 1:
        raise Exception('len of variables inputed must be 1')
    try:
        return [k for k, v in locals().items() if v is variable[0]][0]
    except:
        return [k for k, v in globals().items() if v is variable[0]][0]


def train_classifiers(dataset_name, features_ID, features,
                      image_objs, mode, retrain=False):

    factor_IDs, unique_factor_IDs = generate_factors(
        dataset_name, image_objs, mode
    )

    factor_hash = hashlib.sha256(
        ''.join([x.ID.GTEx_ID for x in image_objs]).encode('utf-8')
    ).hexdigest()

    filepath = CACHE_PATH + features_ID \
                          + f'_{mode}_{factor_hash}_classifiers.pkl'

    X_train, X_test, y_train, y_test = train_test_split(
        features, factor_IDs,
        stratify=factor_IDs,
        test_size=0.25, random_state=42
    )
    if isfile(filepath) and not retrain:
        logger.debug(f'Loading {mode} classifiers from cache')
        lr, svm, rf = pickle.load(open(filepath, 'rb'))
    else:
        logger.debug('Training classifier')

        lr = LogisticRegression()
        svm = SVC(probability=True)
        rf = RandomForestClassifier()

        lr.fit(X_train, y_train)
        svm.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        pickle.dump([lr, svm, rf], open(filepath, 'wb'))

    lr_score = lr.score(X_test, y_test)
    svm_score = svm.score(X_test, y_test)
    rf_score = rf.score(X_test, y_test)

    print(
        (
            f'{mode} label classifer accuracy. '
            f'n_classes={len(unique_factor_IDs)}'
        )
    )

    print(f'n_train: {len(X_train)}, n_test: {len(X_test)}')
    print(
        (
            f'LR: {lr_score:.3f}\n'
            f'SVM: {svm_score:.3f}\n'
            f'RF: {rf_score:.3f}'
        )
    )
    return lr_score, svm_score, rf_score


def aggregate_features(dataset_name, features, image_objs, mode, aggregation):
    logger.debug('Aggregating features')

    factor_IDs, unique_factor_IDs = generate_factors(
        dataset_name, image_objs, mode
    )

    n_factors = len(unique_factor_IDs)
    aggregated_features = np.zeros((n_factors, features.shape[1]))
    aggregated_image_objs = []

    for f in np.unique(factor_IDs):
        f_idx = factor_IDs == f
        f_image_objs = np.array(image_objs)[f_idx].tolist()
        f_features = features[f_idx]
        aggregated_feature = aggregation(f_features, 0)
        aggregated_features[f] = aggregated_feature
        aggregated_image_objs.append(f_image_objs[0])

    return aggregated_features, aggregated_image_objs


def generate_factors(dataset_name, image_objs, mode):
    logger.debug('Generating factors')
    factor_hash = hashlib.sha256(
        ''.join([x.ID.GTEx_ID for x in image_objs]).encode('utf-8')
    ).hexdigest()

    filepath = CACHE_PATH + f'{mode}_factors_{factor_hash}'
    if isfile(filepath):
        logger.debug('Loading factor IDs from cache')
        factor_IDs, unique_factor_IDs = pickle.load(open(filepath, 'rb'))
    else:
        logger.debug('Generating factor IDs')
        if mode == 'tissue_IDs':
            tissue_IDs = get_tissue_IDs(dataset_name, image_objs, cache=False)
            factor_IDs, unique_factor_IDs = pd.factorize(tissue_IDs)
        elif mode == 'GTEx_IDs':
            GTEx_IDs = [image.ID.GTEx_ID for image in image_objs]
            factor_IDs, unique_factor_IDs = pd.factorize(GTEx_IDs)
        pickle.dump([factor_IDs, unique_factor_IDs], open(filepath, 'wb'))

    return factor_IDs, unique_factor_IDs


def subselect_tissue(dataset_name, tissue, features, image_objs):
    tissue_IDs = get_tissue_IDs(dataset_name, image_objs)
    tissue_idx = tissue_IDs == tissue
    tissue_features = features[tissue_idx, :]
    tissue_image_objs = np.array(image_objs)[tissue_idx].tolist()
    return tissue_features, tissue_image_objs


@click.command()
@click.option(
    '--n_images', default=10,
    help="Number of images per tissue"
)
@click.option(
    '--n_tissues', default=6,
    help="Number of tissues with most numbers of samples"
)
@click.option(
    '--n_patches', default=50,
    help="Number of patches to sample"
)
@click.option(
    '--patch_size', default=128,
    help="Size of patches to sample in pixels"
)
@click.option(
    '--model_file_id',
    default='CA_ps128_n12000_e500_lr0.0001_bs64_dim256_do0.5',
    help="Model file ID to generate features from"
)
def main(n_images, n_tissues, n_patches, patch_size, model_file_id):
    logger.info('Initializing cluster_classify script')
    dataset = Dataset(n_tissues=n_tissues, n_images=n_images)
    data = dataset.sample_data(patch_size, n_patches)
    patches, GTEx_IDs = data
    image_objs = [Image(x) for x in GTEx_IDs]

    dataset_name = ''.join([s for s in str(dataset) if s.isalnum()])
    features_ID = dataset_name + f'_{n_patches}_{patch_size}_{n_images}' \
                               + model_file_id

    features = generate_features(features_ID, patches, model_file_id)

    a_features, a_image_objs = aggregate_features(
        dataset_name, features, image_objs, 'GTEx_IDs', np.mean
    )

    a_features, a_image_objs = aggregated_features['GTEx_factor_IDs']['np.mean']

    lung_features, lung_image_objs = subselect_tissue(
        dataset_name, 'Lung', features, image_objs
    )

    train_classifiers(
        dataset_name, features_ID, lung_features, lung_image_objs, 'GTEx_IDs',
        retrain=True
    )


    # plot_PCA(dataset_name, a_features, a_image_objs, 'tissue_IDs')

    # plt.show()


if __name__ == '__main__':
    logging.basicConfig(
        filename='logs/cluster_classify.log',
        level=logging.DEBUG,
        format=(
            "%(asctime)s | %(name)s | %(processName)s |"
            "%(levelname)s: %(message)s"
        )
    )
    main()
