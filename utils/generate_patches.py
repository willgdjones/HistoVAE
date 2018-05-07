import argparse
import numpy as np
import glob
import h5py
from tqdm import tqdm

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tissue', help='All or <tissue_type>')
parser.add_argument('-s', '--nsamples', help='Number of images per tissue type')
parser.add_argument('-p', '--npatches', help='Number of patches per image')
parser.add_argument('-z', '--patchsize', help='Patchsize to take')
args = vars(parser.parse_args())

all_tissues = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

npatches = int(args['npatches'])
nsamples = int(args['nsamples'])
patchsize = int(args['patchsize'])
if args['tissue'] == 'all':
    tissues = all_tissues
else:
    assert args['tissue'] in all_tissues
    tissues = [args['tissue']]



labels = []
patches = []


for tissue in tissues:
    choices = np.random.choice(glob.glob(GTEx_directory + '/data/patches/{}/*{}*'.format(tissue,patchsize)), nsamples)
    print (tissue)

    for sample_choice in tqdm(choices):
        with h5py.File(sample_choice) as g:
            p = g['/patches'].value
            idx = np.random.choice(range(p.shape[0]), npatches)
        patches.extend(p[idx,:,:,:])
        labels.extend([tissue]*len(idx))



if args['tissue'] == 'all':
    tissues = 'all'

output_file = 'patches-{tissues}_s-{nsamples}_p-{npatches}.hdf5'.format(tissues=tissues,
                                                                    nsamples=nsamples,
                                                                    npatches=npatches)

output_path = GTEx_directory + '/data/h5py/' + output_file
with h5py.File(output_path,'w') as f:
    f.create_dataset('/patches', data=patches)
    f.create_dataset('/labels', data=[x.encode('utf-8') for x in labels])
