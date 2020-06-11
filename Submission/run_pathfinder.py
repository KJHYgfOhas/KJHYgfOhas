import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import numpy as np
import h5py
from saliency import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import *
from BilinearInterpolate3D import BilinearInterpolate3D
from OrthogonalConv import NetXCycle
import os


gpu_memory_fraction = 0.4
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

BATCH_SIZE = 16

def pathfinder_main():
    pathfinder_model_path = 'Pathfinder/models/orthogonal_models/orthogonal_pathfinder14_kernelsize20'
    untar = not bool(pathfinder_model_path.find('.h5')+1)

    if untar:
        os.system('sh untar.sh ' + pathfinder_model_path)

        
    if untar:
        a = os.popen('ls ' + pathfinder_model_path)
        a = a.read()
        a = a.split('\n')

    if untar:
        model = [i for i in a if i.find('.h5') != -1][0]
    
    if untar:
        pathfinder_model_path = pathfinder_model_path + '/' + model

        pathfinder_model = load_model(pathfinder_model_path, custom_objects={'BilinearInterpolate3D':BilinearInterpolate3D, 'NetXCycle': NetXCycle})

    if untar:
        os.system('rm ' + pathfinder_model_path)

    pathfinder_model.summary()

    pathfinder_dataset_path = 'Pathfinder/data/pathfinder_path14_test.h5'

    # Load dataset
    x_test, y_test = read_h5_dataset(pathfinder_dataset_path, scale = False)

    # Evaluating 
    scores = pathfinder_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    pathfinder_main()