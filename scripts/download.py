import sys
import requests.packages.urllib3
from multiprocessing import Process, Pool
requests.packages.urllib3.disable_warnings()
sys.path.append('.')
from src.models import Sample, Image, Donor, Annotations, Collection

# images = Collection.images
lung_samples = [
    s for s in Collection.samples
    if (s.tissue == 'Lung' and s.has_image())
]

himage = [s.has_image() for s in Collection.samples]
# all_samples = Annotations.sample.apply(Sample, axis=1)
# image = Donor('1117F').get_samples().iloc[7].get_image()
import pdb; pdb.set_trace()
