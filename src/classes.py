import pandas as pd
import logging
import requests
from os.path import isfile

logger = logging.getLogger(__name__)

TXT_PATH = 'txt/'
IMAGE_PATH = 'data/images/'


class Image():
    def __init__(self, imageID):
        self.ID = ID(imageID)

    def __repr__(self):
        return f"<Image:{self.ID.donor}-{self.ID.sample}>"

    def load(self):
        raise NotImplementedError

    def download(self):
        imageID = f'GTEX-{self.ID.donor}-{self.ID.sample}'
        output_filename = imageID + ".svs"
        output_filepath = IMAGE_PATH + output_filename

        if isfile(output_filepath):
            logger.debug(f'{str(self)} imagefile exists')
            return True
        else:
            logger.debug(f'Downloading {str(self)}')
            session = requests.session()
            URL = "https://brd.nci.nih.gov/brd/imagedownload/" + imageID
            response = session.get(URL)
            if response.ok:
                with open(output_filepath, 'wb') as outfile:
                    outfile.write(response.content)
                return True
            else:
                logger.debug(f'Something wrong with {str(self)}')
                return False



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
