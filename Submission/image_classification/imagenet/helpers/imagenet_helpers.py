from __future__ import print_function
from tensorflow.keras.metrics import top_k_categorical_accuracy

def top_1_accuracy(y_true, y_pred):
    """
        Calculates top-1 accuracy of the predictions.
        To be used as evaluation metric in model.compile().

        Arguments:
            y_true -- array-like, true labels
            y_pred -- array-like, predicted labels

        Returns:
            top-1 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

def top_5_accuracy(y_true, y_pred):
    """
        Calculates top-5 accuracy of the predictions.
        To be used as evaluation metric in model.compile().

        Arguments:
            y_true -- array-like, true labels
            y_pred -- array-like, predicted labels

        Returns:
            top-1 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


import h5py

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

class MultiGPUCheckpoint(ModelCheckpoint):    
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model
