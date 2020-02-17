import h5py
import numpy as np

CENTRALIZING_CONST = 120.84449672851562
NORMALIZING_CONST = 64.13596441053241
QUANTIZATION_SIZE = 255

def read_h5_dataset(filename, scale = True):
    'Reads a h5 dataset of images and returns it. Images are scaled accordinhg to CIFAR10 statistics.'
    with h5py.File(filename, 'r') as h5File:
        if 'dataset' not in h5File.keys(): raise ValueError('h5 file has no dataset')
        dataset = h5File['dataset']
        if scale:
            x_set = (np.array(dataset['x_set'])*QUANTIZATION_SIZE-CENTRALIZING_CONST)/NORMALIZING_CONST
        else:
            x_set = np.array(dataset['x_set'])
        y_set = np.array(dataset['y_set'])
    return x_set, y_set