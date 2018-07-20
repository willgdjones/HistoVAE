import pandas as pd
import logging
import requests
import cv2
import mahotas
from os.path import isfile
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from tqdm import tqdm
from tables import open_file, Atom


logger = logging.getLogger(__name__)

TXT_PATH = 'txt/'
IMAGE_PATH = 'data/images/'
PATCH_PATH = 'data/patches/'


class Image():
    def __init__(self, imageID):
        self.imageID = imageID
        self.ID = ID(imageID)
        self.imagefilepath = IMAGE_PATH + self.imageID + ".svs"
        self.patchfilepath = PATCH_PATH + self.imageID + ".svs"

    def __repr__(self):
        return f"<Image:{self.ID.donor}-{self.ID.sample}>"

    def load_slide(self):
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
            response = session.get(URL)
            if response.ok:
                with open(self.filepath, 'wb') as outfile:
                    outfile.write(response.content)
                return True
            else:
                logger.debug(f'Something wrong with {str(self)}')
                return False

    def has_patches(self):
        raise NotImplementedError

    def get_patches(self):
        h5file = open_file(
            self.patchfilepath, mode='w', title=f'{self.imageID} patches'
        )
        atom = Atom.from_dtype(np.dtype('int8'))

        for patchsize in [128, 256]:

            logger.debug(
                f'Retrieving patches for {self.imageID} patchsize: {patchsize}'
            )
            slide = self.load_slide()

            dslevel = slide.level_count - 1
            maxcoord = Coord(*slide.level_dimensions[0])
            dscoord = Coord(*slide.level_dimensions[-1])

            downsample = slide.level_downsamples[-1]

            logger.debug('Reading region')
            dsregion = np.array(
                slide.read_region((0, 0), dslevel, (dscoord.y, dscoord.x))
            )

            logger.debug('Performing GaussianBlur')
            blurreddsregion = cv2.GaussianBlur(dsregion, (51, 51), 0)
            blurreddsregion = cv2.cvtColor(
                blurreddsregion, cv2.COLOR_BGR2GRAY
            )
            T_otsu = mahotas.otsu(blurreddsregion)

            mask = np.zeros_like(blurreddsregion)
            mask[blurreddsregion < T_otsu] = 255

            # Downsampled Patch Size
            dsps = np.round(patchsize / downsample).astype(int)

            limitcoord = maxcoord / patchsize

            logger.debug('Computing mask coordinates')

            coords = [
                Coord(int(dsps/2 + i*dsps), int(dsps/2 + j*dsps))
                for i in range(limitcoord.x) for j in range(limitcoord.y)
            ]

            mask_coords = [
                coord * downsample
                for coord in coords
                if mask[coord.y, coord.x] > 0
            ]

            tile_generator = DeepZoomGenerator(
                slide, tile_size=patchsize, overlap=0, limit_bounds=False
            )
            level_count = tile_generator.level_count - 1

            logger.debug('Retrieving tiles')


            logger.debug(
                f'Saving patches for {self.imageID} patchsize: {patchsize}'
            )

            N = len(mask_coords)

            carray = h5file.create_carray(
                '/', f'Size {patchsize}', atom,
                (patchsize, patchsize, 3, N)
            )

            tiles = np.zeros((patchsize, patchsize, 3, N))

            for (i, coord) in tqdm(enumerate(mask_coords)):
                tiles[:, :, :, i] = np.array(
                    tile_generator.get_tile(
                        level_count, (coord.x / patchsize, coord.y / patchsize)
                    )
                )
            carray[:,:,:,:] = tiles



class Coord():
    def __init__(self, y, x):
        self.x = x
        self.y = y

    def __mul__(self, other):
        return Coord(
            np.round(self.y*other).astype(int),
            np.round(self.x*other).astype(int)
        )

    def __truediv__(self, other):
        assert other != 0
        return Coord(
            np.round(self.y / other).astype(int),
            np.round(self.x / other).astype(int)
        )


    def __repr__(self):
        return f"<Coord:X{self.x}Y{self.y}>"

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
        return f"Sample:{self.tissue[:5]}|ID:{self.ID.donor}-{self.ID.sample}-{self.ID.aliquot}"

    def get_image(self):
        return Image(self.imageID)

    def has_image(self):
        return self.imageID in Collection.imageIDs

    def has_expression(self):

        expressionID = f"GTEX-{self.ID.donor}-{self.ID.sample}-SM-{self.ID.aliquot}"

        return expressionID in Collection.expressionIDs

    def get_aliquots(self):
        raise NotImplementedError


class Donor():
    def __init__(self, donorID):
        self.donorID = donorID

    def __repr__(self):
        return f"<Donor:{self.donorID}>"

    def get_samples(self):
        return Annotations.sample[
            Annotations.sample['SAMPID'].apply(
                lambda x: ID(x).donor, 1
            ) == self.donorID
        ].apply(Sample, axis=1)

    def get_genotype(self):
        raise NotImplementedError


class Annotations():
    sample = pd.read_csv(
        TXT_PATH + 'GTEx_v7_Annotations_SampleAttributesDS.txt',
        sep='\t'
    )
    subject = pd.read_csv(
        TXT_PATH + 'GTEx_v7_Annotations_SubjectPhenotypesDS.txt',
        sep='\t'
    )


class Collection():

    samples = Annotations.sample.apply(Sample, axis=1)

    with open(TXT_PATH + 'image_ids.txt') as image_file:
        imageIDs = image_file.read().splitlines()
        images = [Image(x) for x in imageIDs]

    expressionIDs = pd.read_csv(
        'txt/expressionIDs.txt', sep=','
    ).values.flatten()

    @staticmethod
    def where(collection, condition):
        logger.debug('Subsetting collection')
        selection = list(filter(condition, eval(f"Collection.{collection}")))
        return sorted(selection, key=lambda x: x.ID.donor)
