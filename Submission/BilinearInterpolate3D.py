import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Lambda, Permute


class BilinearInterpolate3D(Layer):
    """
    Bi-linearly interpolates over the 
    """
    def __init__(self, output_shape, **kwargs):
        """
        output_shape: shape of the output tensor(this is considered to be channel_last)
        """
        assert(len(output_shape) == 3)
        super(BilinearInterpolate3D, self).__init__(**kwargs)

        self._output_shape = np.array(output_shape)

        # setting up the lambda layer used for bilinear interpolation
        def resize_bilinear(tensor, output_shape): return tf.image.resize_bilinear(tensor, output_shape, align_corners=True)
        self._bilinear_interpolation_x_y = Lambda(resize_bilinear, arguments={'output_shape':self._output_shape[0:2]})
        self._permute_x_z = Permute((3,2,1))
        self._bilinear_interpolation_z_y = Lambda(resize_bilinear, arguments={'output_shape':self._output_shape[1:3][::-1]})
        self._permute_z_x = Permute((3,2,1))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert(len(input_shape) == 4)
        super(BilinearInterpolate3D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # logic executed when the layer is called
        assert(len(x.shape) == 4)
        x = self._bilinear_interpolation_x_y(x)
        x = self._permute_x_z(x)
        x = self._bilinear_interpolation_z_y(x)
        x = self._permute_z_x(x)
        return x 

    def get_config(self):
        # need to override this
        config = super(BilinearInterpolate3D, self).get_config().copy()
        config.update({
           'output_shape': self._output_shape
        })
        return config