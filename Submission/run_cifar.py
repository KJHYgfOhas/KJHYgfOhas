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

gpu_memory_fraction = 0.4
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

BATCH_SIZE = 16


def cifar_main():
    
    cifar_model_path = 'image_classification/models/baseline_models/baseline_cifar10_depth18.h5'
    cifar_dataset_path = 'image_classification/data/CIFAR-10_test.h5'
    cifar_model = load_model(cifar_model_path, custom_objects={'BilinearInterpolate3D':BilinearInterpolate3D})

    opt_rms = RMSprop(lr=0.001,decay=1e-6)

    # Compiling the model
    cifar_model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

    cifar_model.summary()

    # Load dataset
    x_test, y_test = read_h5_dataset(cifar_dataset_path)

    # Evaluating
    scores = cifar_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    image_index = np.random.randint(0, x_test.shape[0]-1)
    image = x_test[image_index]
    plt.imsave('input_image.png', (image * NORMALIZING_CONST + CENTRALIZING_CONST)/QUANTIZATION_SIZE)

    no_of_nodes = 3
    layer_list = [2,70]
    saliency = get_saliency_maps_and_fitted_ellipses(cifar_model, layer_list, no_of_nodes, image)

    save_images(*saliency)



if __name__ == '__main__':
    cifar_main()