import tensorflow as tf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax

def VisualizeImageGrayscale(image_3d):
    '''
    Returns a 3D tensor as a grayscale 2D tensor by summing the absolute values over the last axis and scaling the results to the interval [0,1].
    :param image3d: the 3d tensor which is to be grey scaled. A 3d numpy array.
    :return: numpy array of dimension shape_of_first_two_axes_of_image_3d that contains the 2d grey_scale image.
    '''

    image_2d = np.sum(np.abs(image_3d), axis=2)
    vmax = np.max(image_2d)
    vmin = np.min(image_2d)
    if vmax == vmin:
        return np.zeros(image_2d.shape)
    return (image_2d - vmin) / (vmax - vmin)

def get_saliency_maps_and_fitted_ellipses(model, layers_index_list, no_of_nodes, image):
    gradients = get_gradients_at_layers(model, layers_index_list, no_of_nodes, image)
    reshaped_grads = np.reshape(gradients, (gradients.shape[0] * gradients.shape[1],) + gradients.shape[2:])

    saliency_map = np.empty(reshaped_grads.shape[0:-1])
    
    for i, gradient in enumerate(reshaped_grads):
        print('iteration', i)
        saliency_map[i] = VisualizeImageGrayscale(gradient) 
    
    print('the shape is ', saliency_map.shape)
    ellipses = plot_ellipse(saliency_map)
    saliency_map = np.reshape(saliency_map, gradients.shape[0:-1])
    ellipses = np.reshape(ellipses, gradients.shape[0:2])
    return saliency_map, ellipses


def get_saliency_stats(model, layer, no_of_nodes, images_batch):
    saliency_map = np.empty((images_batch.shape[0] * no_of_nodes,) + images_batch.shape[1:-1])
    
    for i, image in enumerate(images_batch):
        image_gradients = np.squeeze(get_gradients_at_layers(model, [layer], no_of_nodes, image))
        print('starting image ', i)
        for j, grad in enumerate(image_gradients):
            print('starting iteration', j)
            saliency_map[i * no_of_nodes + j] = VisualizeImageGrayscale(grad) 
    
    means, eigen_values, _ = generate_stats_for_multiple_saliencies(saliency_map)
    not_empty_grads = np.all(np.logical_not(np.isnan(means)), axis = 1)
    print(not_empty_grads)

    eigen_values = eigen_values[not_empty_grads]
    return np.sqrt(np.sum(eigen_values, axis =1))

def saliency_statistics(saliency):
    '''
    Calculates the statistics of a non-negative matrix. The matrix is assumed to be non-normalized density values of a 2 dimensional
    random vector taking values in [0,...,matrix_rows] x [0,..., matrix_cols]. The procedure normalizes the density and returns the means of the
    random variable, covariance matrix and eigenvalues of the covariance matrix.
    :param saliency: a non-negative matrix (2-dim numpy array).
    :return: mean of the random variable associated to the matrix, covariance and eigenvalues of the covariance matrix.
    '''
    if saliency.ndim != 2:
        raise ValueError('the saliency map is of incompatible dimensions. Should be a two dim array')
    total_sum = np.sum(saliency)
    if total_sum == 0:
        raise ValueError('the saliency map is constant')
    saliency = saliency/total_sum
    mean_x = mean_y = var_x = var_y = cov_xy = 0

    for i in range(saliency.shape[0]):
        for j in range(saliency.shape[1]):
            mean_x += saliency[i,j] * j
            mean_y += saliency[i,j] * i

    for i in range(saliency.shape[0]):
        for j in range(saliency.shape[1]):
            var_x += np.power(j - mean_x, 2) * saliency[i,j]
            var_y += np.power(i - mean_y, 2) * saliency[i,j]
            cov_xy += j * i * saliency [i,j]

    cov_xy = cov_xy - mean_x * mean_y
    cov_xy = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype = 'float32')
    eigen_values, eigen_vectors = np.linalg.eigh(cov_xy)
    return mean_x, mean_y, cov_xy, eigen_values, eigen_vectors

def generate_stats_for_multiple_saliencies(saliency_maps):
    eigen_values = np.empty((saliency_maps.shape[0], 2))
    means = np.empty((saliency_maps.shape[0], 2))
    eigen_vectors = np.empty((saliency_maps.shape[0], 2, 2))

    eigen_values[:] = np.nan
    means[:] = np.nan
    eigen_vectors[:] = np.nan
    number_of_constant_pics = 0

    for i in range(saliency_maps.shape[0]):
        try:
            mean_x, mean_y, cov_xy, eigh, eig_vecs = saliency_statistics(saliency_maps[i])
            eigen_values[i] = eigh
            eigen_vectors[i] = eig_vecs
            means[i] = [mean_x, mean_y]
        except ValueError:
            number_of_constant_pics += 1
    print('number of constant pics:', number_of_constant_pics)
    return means, eigen_values, eigen_vectors

def plot_ellipse(grey_scale_saliency, scale_in_std = 2):

    ellipses = np.empty((grey_scale_saliency.shape[0]), dtype = Ellipse)
    ellipses[:] = None

    means, eigen_values, eigen_vectors = generate_stats_for_multiple_saliencies(grey_scale_saliency)
    scaled_eigen_values = 2 * np.sqrt(eigen_values) * np.sqrt(scale_in_std)

    count = 0
    for index in range(eigen_values.shape[0]):
        if not np.all(np.isnan(means[index])):
            angle = np.degrees(np.arctan2(eigen_vectors[index, 1, 1], eigen_vectors[index, 1, 0]))
            if (angle != 0):
                count += 1
            ellipses[index] = Ellipse(xy=means[index], width=scaled_eigen_values[index, 1], \
                                      height=scaled_eigen_values[index, 0], angle=angle, fill=False, \
                                                  linewidth=2, edgecolor='yellow')
    print('Number of nonzero angles:', count)
    return ellipses

def get_gradients_at_node(model, node, x, learning_phase=0.):
    '''
    :param model: the model to be evaluated, an instance of keras.model.Model
    :param node:  a scalar tensor.
    :param x: an input to model (the point at which the gradient is evaluated), without batch dimension.
    :param learning_phase: 0 or 1 depending on whether inference should be in training mode or not.
    :return: numpy array of dimension 1 * x.shape, that contains the gradient dy/dx (x), where y is node, x is model inputs and x is the point x at which the gradient is evaluated
    '''
    inputs = model.input
    grad = tf.gradients(node, inputs)[0]
    func = K.function([inputs, K.learning_phase()], grad)
    return np.array(func([np.expand_dims(x, 0), learning_phase]))

def get_gradients_at_layers(model, layer_index_list, no_of_nodes, image):
    '''
    The function get a list of layers, samples a number of nodes from each of the layers and returns the gradients of these nodes with respect to the input layer of the model evaluated on image.
    :param model: the model on which the gradients are evaluated - an instance of tensorflow.keras.models.Model
    :param layer_index_list: a list of integers - specifies the indices of layers.
    :param no_of_nodes: a positive integer - specifies the number of nodes to be sampled from each of the layers.
    :param image: the input image on which the gradients are to be evaluated.
    :return: a numpy array of shape (no_of_layers, no_of_nodes, shape_of_inputs_to_the_model) that contains the calculated gradients.
    '''

    #print(len(layer_index_list))
    input_layer = model.input
    shape = (len(layer_index_list), no_of_nodes,) + K.int_shape(input_layer)[1:]

    saliency_map = np.empty(shape = shape)
    saliency_map[:] = np.nan

    for i, layer in enumerate(layer_index_list):
        output_layer = K.flatten(model.layers[layer].output[0, :])
        no_of_outputs = K.int_shape(output_layer)[0]
        random_nodes = np.random.randint(low = 0, high = no_of_outputs, size = no_of_nodes)

        for j, node_index in enumerate(random_nodes):
            node = output_layer[node_index]
            saliency_map[i,j] = get_gradients_at_node(model, node, image)
    return saliency_map

'''
tf.compat.v1.disable_eager_execution()

input_tensor = Input(shape = (2))
const = tf.constant([[2,0], [0,1]], dtype = 'float32', shape = (2,2))
res = tf.matmul(input_tensor,const)

model = Model(inputs = input_tensor, outputs = res)
res = softmax(res)
model_with_soft_max = Model(inputs = input_tensor, outputs = res)

true_labels = [1, 0]
x = [1, 1]

node = model_with_soft_max.layers[2].output[0,1]
saliency = get_gradients_at_node(model, node ,np.array(x))
print(saliency)
assert(np.array_equal(saliency, np.array([[2.0, 0.0]])))
print(saliency)
'''

'''
true_labels = [[1, 0], [0, 1]]
x = [[1, 1], [1, 3]]

saliency = get_saliency_from_predicted_classes(self._model, x, true_labels)
assert(np.array_equal(np.array(saliency), [[2.0, 0.0], [0.0, 1.0]]))
'''