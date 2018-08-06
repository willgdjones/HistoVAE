import pandas as pd
import logging
import requests
import cv2
import mahotas
import pickle
from os.path import isfile
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from tqdm import tqdm
from tables import open_file, Atom, Filters
from collections import Counter
from itertools import cycle

logger = logging.getLogger(__name__)

TXT_PATH = 'txt/'
IMAGE_PATH = 'data/images/'
PATCH_PATH = 'data/patches/'
CACHE_PATH = '.cache/'


class Image():
    def __init__(self, imageID):
        self.imageID = imageID
        self.ID = ID(imageID)
        self.imagefilepath = IMAGE_PATH + self.imageID + ".svs"
        self.patchcoordsfilepath = PATCH_PATH + self.imageID + ".hdf5"
        self.patchcoordsfile = None

    def __repr__(self):
        return f"<Image:{self.ID.donor}-{self.ID.sample}>"

    def has_sample(self):
        return self.imageID in set(Collection.sample_imageIDs)

    def get_sample(self):
        return Collection.where(
            'samples', lambda s: s.imageID == self.imageID
        )[0]

    def get_slide(self):
        logger.debug(f'Loading {self.imagefilepath}')
        return open_slide(self.imagefilepath)

    def is_downloaded(self):
        return isfile(self.imagefilepath)

    def download(self):
        if self.is_downloaded():
            logger.debug(f'{str(self)} is downloaded')
            return True
        else:
            logger.debug(f'Downloading {str(self)}')
            session = requests.session()
            URL = "https://brd.nci.nih.gov/brd/imagedownload/"\
                    + self.imageID
            logger.debug(f'Getting URL for {str(self)}')
            response = session.get(URL, stream=True)
            if response.ok:
                outfile = open(self.imagefilepath, 'wb')
                # total_length = int(response.headers.get('content-length'))
                total_length = 500 * ((2**10)**2)
                chunk_size = 1024
                expected_size = (total_length/1024) + 1
                response_iter = response.iter_content(chunk_size=chunk_size)
                for chunk in tqdm(response_iter, total=expected_size):
                    if chunk:
                        outfile.write(chunk)
                        outfile.flush()
                logger.debug(f'Successfully downloaded {str(self)}')
                outfile.close()
                return True
            else:
                logger.debug(f'Something wrong with {str(self)}')
                return False

    def has_patchcoords(self):
        return isfile(self.patchcoordsfilepath)

    def get_patchcoordsfile(self):
        logger.debug(f'Retrieving patches for {self.imageID}')
        patchcoordsfile = open_file(
            self.patchcoordsfilepath, mode='r'
        )
        return patchcoordsfile

    def generate_patchcoords(self):
        if self.has_patches():
            patchcoordsfile = self.get_patchcoordsfile()
            return patchcoordsfile

        logger.debug(
            f'Generating patches for {self.imageID}'
        )
        filters = Filters(complib='zlib', complevel=5)
        patchcoordsfile = open_file(
            self.patchcoordsfilepath, mode='w',
            title=f'{self.imageID} patches',
            filters=filters
        )

        atom = Atom.from_dtype(np.dtype('uint16'))
        slide = self.get_slide()
        dslevel = slide.level_count - 1
        dscoord = Coord(*slide.level_dimensions[-1])
        downsample = slide.level_downsamples[-1]

        logger.debug('Reading region')
        dsregion = np.array(
            slide.read_region((0, 0), dslevel, (dscoord.x, dscoord.y))
        ).transpose(1, 0, 2)

        logger.debug('Performing GaussianBlur')
        blurreddsregion = cv2.GaussianBlur(dsregion, (51, 51), 0)
        blurreddsregion = cv2.cvtColor(
            blurreddsregion, cv2.COLOR_BGR2GRAY
        )
        T_otsu = mahotas.otsu(blurreddsregion)
        mask = np.zeros_like(blurreddsregion)
        mask[blurreddsregion < T_otsu] = 1

        for patchsize in [128, 256, 512, 1024]:

            logger.debug(
                f'patchsize: {patchsize}'
            )
            dsps = np.round(patchsize / downsample).astype(int)
            limitcoord = Coord(*dsregion.shape[:2]) / dsps

            logger.debug('Computing downsampled centers')
            dscentercoords = [
                Coord(
                    int(dsps/2 + i*dsps),
                    int(dsps/2 + j*dsps)
                )
                for i in range(limitcoord.x - 1)
                for j in range(limitcoord.y - 1)
            ]
            logger.debug('Computing mask coordinates')

            assert (
                dscentercoords[-1].x < mask.shape[0] and
                dscentercoords[-1].y < mask.shape[1]
            )

            mask_centers = list(filter(
                lambda c: mask[c.x, c.y] == 1,
                dscentercoords
            ))
            tile_generator = DeepZoomGenerator(
                slide, tile_size=patchsize, overlap=0, limit_bounds=False
            )
            level_count = tile_generator.level_count - 1

            logger.debug(
                f'Saving patches for {self.imageID} patchsize: {patchsize}'
            )

            N = len(mask_centers)
            # tiles = np.zeros((patchsize, patchsize, 3, N), dtype=np.uint8)
            logger.debug('Retrieving tiles')

            assert (mask_centers[-1] * downsample) / patchsize <\
                Coord(*tile_generator.level_tiles[-1])

            valid_coords = []
            for (i, coord) in tqdm(list(enumerate(mask_centers))):
                tile = np.array(
                    tile_generator.get_tile(
                        level_count, (
                            (downsample * coord.x) / patchsize,
                            (downsample * coord.y) / patchsize)
                    )
                )
                if ((tile > T_otsu).sum() / np.prod(tile.shape)) < 0.25:
                    valid_coords.append(coord)
            n = len(valid_coords)
            valid_coords = np.array(
                [c.to_array() for c in valid_coords]
            )
            logger.debug((
                f"Selected {n} tiles out of {N} ({n/N:0.2})"
                "with percent whitespace < 0.25"
            ))
            if n > 0:
                carray = patchcoordsfile.create_carray(
                    '/', f'Size{patchsize}', atom,
                    (n, 2)
                )
                carray[:, :] = valid_coords
        patchcoordsfile.close()
        return True

    def get_patches(self, s, n):
        """Generate a set of patches with size s and of length n"""
        patchcoordsfile = self.get_patchcoordsfile()\
            if not self.patchcoordsfile else self.patchcoordsfile
        coords = [
            Coord(*c) for c in patchcoordsfile.get_node(
                f'/Size{s}'
            ).read()
        ]
        replace = len(coords) < n
        coords_choice = np.random.choice(coords, n, replace=replace)
        slide = self.get_slide()
        patches = generate_patches(slide, coords_choice, s)
        return patches


def generate_patches(slide, coords, s):

    n = len(coords)
    patches = np.zeros((n, s, s, 3), dtype=np.float16)
    downsample = slide.level_downsamples[-1]

    for (i, coord) in enumerate(coords):
        tile_generator = DeepZoomGenerator(
            slide, tile_size=s, overlap=0, limit_bounds=False
        )
        level_count = tile_generator.level_count - 1
        tile = tile_generator.get_tile(
            level_count, (
                (downsample * coord.x) / s,
                (downsample * coord.y) / s)
        )
        patch = np.array(tile)
        ppatch = process(patch)
        patches[i, :, :, :] = ppatch
    return patches


def process(image):
    s = image.shape[0]
    pimage = np.reshape(image, (1, s, s, 3))
    pimage = ((255 - pimage) / 255).astype(np.float16)
    return pimage


def deprocess(pimage):
    image = np.squeeze(pimage)
    image = 255 - (image * 255).astype(np.uint8)
    return image


class Coord():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other):
        return Coord(
            np.round(self.x * other).astype(int),
            np.round(self.y * other).astype(int)
        )

    def __truediv__(self, other):
        assert other != 0
        return Coord(
            np.round(self.x / other).astype(int),
            np.round(self.y / other).astype(int)
        )

    def __repr__(self):
        return f"<Coord:X{self.x}Y{self.y}>"

    def __lt__(self, other):
        return (self.x < other.x and self.y < other.y)

    def to_array(self):
        return np.array([self.x, self.y])


class ID():
    def __init__(self, GTExID):
        split = GTExID.split('-')
        if GTExID.startswith('K-562'):
            self.aliquot = split[-1]
            self.donor = 'K-562'
            self.sample = 'K-562'
        else:
            self.GTExID = GTExID
            self.donor = split[1]
            self.sample = split[2]
        self.aliquot = split[-1] if len(split) == 5 else None

    def __repr__(self):
        return f"ID:{self.GTExID}"


class Sample():
    """
        Defines the Sample object.
    """

    def __init__(self, row):
        self.ID = ID(row['SAMPID'])
        self.imageID = f'GTEX-{self.ID.donor}-{self.ID.sample}'
        self.tissue = row['SMTSD']
        self.annotations = row['SMPTHNTS']

    def __repr__(self):
        return (
            f"Sample:{self.tissue[:5]}|"
            f"ID:{self.ID.donor}-{self.ID.sample}-{self.ID.aliquot}"
        )

    def get_image(self):
        return Image(self.imageID)

    def has_image(self):
        return self.imageID in Collection.imageIDs

    def has_expression(self):

        expressionID = (
            f"GTEX-{self.ID.donor}-{self.ID.sample}-SM-{self.ID.aliquot}"
        )

        return expressionID in Collection.expressionIDs

    def get_aliquots(self):
        raise NotImplementedError


class Donor():
    def __init__(self, donorID):
        self.donorID = donorID

    def __repr__(self):
        return f"<Donor:{self.donorID}>"

    def get_samples(self):
        return Annotation.samples[
            Annotation.samples['SAMPID'].apply(
                lambda x: ID(x).donor, 1
            ) == self.donorID
        ].apply(Sample, axis=1)

    def get_genotype(self):
        raise NotImplementedError


class Annotation():
    samples = pd.read_csv(
        TXT_PATH + 'GTEx_v7_Annotations_SampleAttributesDS.txt',
        sep='\t'
    )
    subjects = pd.read_csv(
        TXT_PATH + 'GTEx_v7_Annotations_SubjectPhenotypesDS.txt',
        sep='\t'
    )


class Collection():
    samples = Annotation.samples.apply(Sample, axis=1)
    sample_imageIDs = [s.imageID for s in samples]

    with open(TXT_PATH + 'image_ids.txt') as image_file:
        imageIDs = image_file.read().splitlines()
        images = [Image(x) for x in imageIDs]

    expressionIDs = pd.read_csv(
        'txt/expressionIDs.txt', sep=','
    ).values.flatten()

    def get_images_with_samples():
        filepath = CACHE_PATH + 'images_with_samples.py'
        logger.debug('Loading images with samples from cache')
        if isfile(filepath):
            images_with_samples = pickle.load(open(filepath, 'rb'))
            return images_with_samples
        else:
            logger.debug('Retrieving images_with_samples')
            images_with_samples = []
            for image in tqdm(Collection.images):
                if image.has_sample():
                    sample = image.get_sample()
                    images_with_samples.append(sample)
            pickle.dump(images_with_samples, open(filepath, 'wb'))
            return images_with_samples

    images_with_samples = get_images_with_samples()

    @staticmethod
    def where(collection, condition):
        selection = list(filter(condition, eval(f"Collection.{collection}")))
        return sorted(selection, key=lambda x: x.ID.donor)


class ToyData():
    tissue_counts = Counter(
        map(lambda x: x.tissue, Collection.images_with_samples)
    ).most_common(6)
    T = len(tissue_counts)

    k = 10
    images = {}
    for tissue, count in tissue_counts:
        tissue_samples = Collection.where(
            'samples', lambda s, tissue=tissue: (
                s.tissue == tissue and
                s.has_image() and
                s.has_expression()
            )
        )
        tissue_images = [x.get_image() for x in tissue_samples][:k]
        images[tissue] = tissue_images

    @staticmethod
    def download():
        for tissue, images in ToyData.images.items():
            logger.debug(f'Downloading {tissue} images')
            results = [image.download() for image in images]
            assert all(results), "Some images failed to download"

    @staticmethod
    def get_patchcoordfiles():
        for tissue, images in ToyData.images.items():
            logger.debug(f'Generating patches for {tissue}')
            results = [image.get_patch_coords() for image in images]
            assert all(results), "Some patches failed to generate"

    @staticmethod
    def generators(s, n, split=0.2):
        logger.debug(f'Generating patchset for ToyData')
        images = ToyData.images
        train, val = train_val_split(images, split)

        def train_gen():
            for (t, tissue) in enumerate(train.keys()):
                for image in train[tissue]:
                    patches = image.get_patches(s, n)
                    yield patches, patches

        def val_gen():
            for (t, tissue) in enumerate(val.keys()):
                for image in val[tissue]:
                    patches = image.get_patches(s, n)
                    yield patches, patches

        return train_gen, val_gen, split


def train_val_split(data, split):
    train = {}
    val = {}
    for (t, tissue) in enumerate(data.keys()):

        train[tissue] = []
        val[tissue] = []
        images = np.array(data[tissue])
        k = len(images)
        ch = np.floor(k * split).astype(int)
        val_idx = np.random.choice(range(k), ch, replace=False)
        train_idx = np.array(list(
            set(range(k)) - set(val_idx)
        ))
        train[tissue].extend(images[train_idx].tolist())
        val[tissue].extend(images[val_idx].tolist())
    return train, val
