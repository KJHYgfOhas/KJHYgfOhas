import tensorflow
from tensorflow.keras.layers import Layer, Conv2D, Permute, Activation, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomUniform

class NetXCycle(Layer):
    def __init__(self, output_shape, kernel_shape=(3,3), activation=lambda x: tensorflow.keras.layers.PReLU()(x), dropout_rate=0.0, kernel_regularizer=None, **kwargs):
        '''
        output_shape shape of the output tensor
        kernel_shape shape of the kernel, defaults to (3,3)
        '''
        assert(len(output_shape) == 3)
        assert(len(kernel_shape) == 2)
        super(NetXCycle, self).__init__(**kwargs)

        self._output_shape = output_shape
        self._kernel_shape = kernel_shape
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._kernel_regularizer = kernel_regularizer

        # input_shape has (batch_size, x, y, z[features])
        # permutation: x y z -> z' x y -> y' z' x -> x' y' z'
        self._conv2D_a = Conv2D(filters=self._output_shape[2],kernel_size=self._kernel_shape,kernel_regularizer=self._kernel_regularizer,padding='same', kernel_initializer = 'he_uniform')
        self._conv2D_b = Conv2D(filters=self._output_shape[1],kernel_size=self._kernel_shape,kernel_regularizer=self._kernel_regularizer,padding='same', kernel_initializer = 'he_uniform')
        self._conv2D_c = Conv2D(filters=self._output_shape[0],kernel_size=self._kernel_shape,kernel_regularizer=self._kernel_regularizer,padding='same', kernel_initializer = 'he_uniform')
        
        self._batch_norm_a = BatchNormalization(axis=3) #, center=False, scale=False)
        self._batch_norm_b = BatchNormalization(axis=1) #, center=False, scale=False)
        self._batch_norm_c = BatchNormalization(axis=2) #, center=False, scale=False)

        # need separate convolutions - we expect weights to differ
        self._dropout_a = Dropout(rate=self._dropout_rate)
        self._dropout_b = Dropout(rate=self._dropout_rate)
        self._dropout_c = Dropout(rate=self._dropout_rate)

        self._permute = Permute((3,1,2))
        self._activation_layer = Activation(self._activation) 


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert(len(input_shape) == 4)

        super(NetXCycle, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):
        assert(len(x.shape) == 4)
        
        x = self._conv2D_a(x)
        x = self._batch_norm_a(x)
        x = self._activation_layer(x)
        x = self._dropout_a(x)
        x = self._permute(x)

        x = self._conv2D_b(x)
        x = self._batch_norm_b(x)
        x = self._activation_layer(x)
        x = self._dropout_b(x)
        x = self._permute(x)

        x = self._conv2D_c(x)
        x = self._batch_norm_c(x)
        x = self._activation_layer(x)
        x = self._dropout_c(x)
        x = self._permute(x)

        return x 

    def get_config(self):
        # need to override this
        config = super(NetXCycle, self).get_config().copy()
        config.update({
           'output_shape': self._output_shape,
           'kernel_shape': self._kernel_shape,
           'activation': self._activation,
           'dropout_rate': self._dropout_rate,
        })
        return config
