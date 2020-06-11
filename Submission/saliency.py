import tensorflow as tf
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax

##########################################################################################################
def visualize_saliency_map(image_3d):
    '''
    Returns a 3D tensor as a grayscale 2D tensor by summing the absolute values over the last axis and scaling the results to the interval [0,1].
    :param image3d: A 3d numpy array of shape (H, W, C).
    :return: numpy array of shape (H, W) that contains the 2d grey_scale image.
    '''

    image_2d = np.sum(np.abs(image_3d), axis=2)
    vmax = np.max(image_2d)
    vmin = np.min(image_2d)
    if vmax == vmin:
        return np.zeros(image_2d.shape)
    return (image_2d - vmin) / (vmax - vmin)

#########################################################################################################
def get_saliency_maps_and_fitted_ellipses(model, layers_index_list, no_of_nodes, image, scale_in_std = 2):
    '''
    The main api procedure to call when fitting covariance error ellipses to saliency maps. It receives a model, the layers
    saliency is to be calculated and the number of nodes to be sampled from each layer. The saliency is to be calculated at the
    sampled nodes and at a given image (received as parameter).
    Input:
        model: keras.models.Model
        layers_index_list: a list of integers, specifying the layers indices in the model.
        no_of_nodes: a positive integer. The number of nodes to be sampled from each layer. 
        image: a numpy array corresponding to the size of the input tensor to the mode, without batch dimension.
        scale_in_std: the scale of the ellipse to be fitted. Corresponds to the number of std deviations in one dim Gaussian fit. 
    Output:
        saliency_map: a numpy array of shape: number_of_layers * no_of_nodes * image_size where image size corresponds to 
            dimensions of the input image to the model, without the channels axis - i.e. grey scale image.
        ellipses: a numpy array of shape number_of_layer * no_of_nodes, where each cell contains a matplotlib.patches.Ellipse
            object - an instance of the ellipse fitted to the corresponding saliency map.   
    '''
    gradients = get_gradients_at_layers(model, layers_index_list, no_of_nodes, image)
    reshaped_grads = np.reshape(gradients, (gradients.shape[0] * gradients.shape[1],) + gradients.shape[2:])

    saliency_map = np.full(reshaped_grads.shape[0:-1], np.nan)
    
    for i, gradient in enumerate(reshaped_grads):
        saliency_map[i] = visualize_saliency_map(gradient) 
    
    ellipses = generate_ellipses(saliency_map, scale_in_std)
    saliency_map = np.reshape(saliency_map, gradients.shape[0:-1])
    ellipses = np.reshape(ellipses, gradients.shape[0:2])
    return saliency_map, ellipses

##########################################################################################################
def get_saliency_stats(model, layer, no_of_nodes, images_batch):
    '''
    The main api function to call to get saliency statistics, i.e the size of the receptive fields. It gets a model and a layer,
    samples a given number of nodes (parameter) from that layer and evaluates the saliency map at these nodes, on all images provided
    in images_batch.
    Input: 
        model: keras.models.Model instance.
        layer: a non negative integer specifying the index of the layer in the model at which saliency is to be evaluated. 
        no_of_nodes: a positive integer - the number of nodes to be sample from the layer. 
        images_batch: a numpy array that contains a batch of images on which saliency is to be evaluated.  
    '''
    saliency_map = np.empty((images_batch.shape[0] * no_of_nodes,) + images_batch.shape[1:-1])
    
    for i, image in enumerate(images_batch):
        image_gradients = np.squeeze(get_gradients_at_layers(model, [layer], no_of_nodes, image))
        for j, grad in enumerate(image_gradients):
            saliency_map[i * no_of_nodes + j] = visualize_saliency_map(grad) 
    
    means, eigen_values, _ = generate_stats_for_multiple_saliencies(saliency_map)
    not_empty_grads = np.all(np.logical_not(np.isnan(means)), axis = 1)
    eigen_values = eigen_values[not_empty_grads]
    return np.sqrt(np.sum(eigen_values, axis =1))

##########################################################################################################
def saliency_statistics(saliency):
    '''
    Calculates the statistics of a non-negative matrix (saliency_map). The matrix is assumed to be non-normalized density values of a 2 dimensional
    random vector taking values in [0,...,matrix_rows] x [0,..., matrix_cols]. The procedure normalizes the density and returns the means of the
    random variable, covariance matrix and eigenvalues of the covariance matrix.
    Input: 
        saliency: a non-negative matrix (2-dim numpy array).
    Output: 
        mean of the random variable associated to the matrix, covariance and eigenvalues of the covariance matrix.
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

##########################################################################################################
def generate_stats_for_multiple_saliencies(saliency_maps):
    '''
    A wrapper around saliency_statistics which allows to get saliency statistics (i.e. receptive field size) for multiple saliencies maps and not just one.
    Input: 
        saliency_maps: a numpy array of shape batch_size * shape of one saliency map of non-negative floats. 
    Output: 
        arrays containing means, eigenvalues and eigenvectors of the density functions inferred from the saliency maps (see
        saliency statistics) for how the underlying density is calculated.
    '''
    eigen_values = np.empty((saliency_maps.shape[0], 2))
    means = np.empty((saliency_maps.shape[0], 2))
    eigen_vectors = np.empty((saliency_maps.shape[0], 2, 2))

    eigen_values[:] = np.nan
    means[:] = np.nan
    eigen_vectors[:] = np.nan

    for i in range(saliency_maps.shape[0]):
        try:
            mean_x, mean_y, cov_xy, eigh, eig_vecs = saliency_statistics(saliency_maps[i])
            eigen_values[i] = eigh
            eigen_vectors[i] = eig_vecs
            means[i] = [mean_x, mean_y]
        except ValueError:
            pass
    return means, eigen_values, eigen_vectors

##########################################################################################################
def generate_ellipses(grey_scale_saliency, scale_in_std = 2):
    '''
        Fit ellipses to a saliency map.
        Input: 
            grey_scale_saliency: a numpy array of dim Number_of_maps x size_of_single_saliency_map
            scale_in_std: the scale of the fitted ellipse - corresponds to the number of standard deviation in 1-dim Gaussian fit
        Output:
            A numpy array of fitted ellipses of size Number_of_maps and type matplotlib.pathes.Ellipse 
    '''
    ellipses = np.empty((grey_scale_saliency.shape[0]), dtype = Ellipse)
    ellipses[:] = None

    centers, eigen_values, eigen_vectors = generate_stats_for_multiple_saliencies(grey_scale_saliency)
    scaled_eigen_values = 2 * np.sqrt(eigen_values) * np.sqrt(scale_in_std)

    for index in range(eigen_values.shape[0]):
        if not np.all(np.isnan(centers[index])):
            angle = np.degrees(np.arctan2(eigen_vectors[index, 1, 1], eigen_vectors[index, 1, 0]))
            ellipses[index] = Ellipse(xy=centers[index], width=scaled_eigen_values[index, 1], \
                                      height=scaled_eigen_values[index, 0], angle=angle, fill=False, \
                                                  linewidth=2, edgecolor='yellow')
    return ellipses

##########################################################################################################
def print_images(saliency_map, ellipses):
    '''
    Receives fitted ellipses and saliency maps and prints the ellipses on top of the saliency maps.
    Input:
        Saliency_map: a numpy array of shape batch_size x no_of_nodes x 2d matrix (grey scale image).
        ellipses: an array of shape batch_size x no_of_nodes filled with matplotlib.patches.Ellipse.
    '''
    fig = plt.figure(figsize = (8,8))
    for i in range(0, saliency_map.shape[0]):
        for j in range(0, saliency_map.shape[1]):
            axes = fig.add_subplot(saliency_map.shape[0], saliency_map.shape[1], saliency_map.shape[1]*i+j+1)
            if ellipses[i,j] is not None:
                axes.add_patch(ellipses[i,j])
            axes.imshow(saliency_map[i,j])

##########################################################################################################

def save_images(saliency_map, ellipses):
    '''
    Receives fitted ellipses and saliency maps and prints the ellipses on top of the saliency maps.
    Input:
        Saliency_map: a numpy array of shape batch_size x no_of_nodes x 2d matrix (grey scale image).
        ellipses: an array of shape batch_size x no_of_nodes filled with matplotlib.patches.Ellipse.
    '''
    fig = plt.figure(figsize = (8,8))
    for i in range(0, saliency_map.shape[0]):
        for j in range(0, saliency_map.shape[1]):
            axes = fig.add_subplot(saliency_map.shape[0], saliency_map.shape[1], saliency_map.shape[1]*i+j+1)
            if ellipses[i,j] is not None:
                axes.add_patch(ellipses[i,j])
            axes.imshow(saliency_map[i,j])
    plt.savefig('saliency.png')

##########################################################################################################

def get_gradients_at_node(model, node, x, learning_phase=0):
    '''
    Gets a keras model and returns the gradient of a node with respect to the input layer at point x
    Input:
        model: the model to be evaluated, an instance of keras.model.Model
        a scalar tensor.
        an input to model (the point at which the gradient is evaluated), without batch dimension.
        learning_phase: 0 or 1 depending on whether inference should be in training mode or not.
    Output: 
        numpy array of dimension 1 * x.shape, that contains the gradient dy/dx (x), where y is node, x is model inputs and x is the point x at which the gradient is evaluated
    '''
    inputs = model.input
    grad = tf.gradients(node, inputs)[0]
    func = K.function([inputs, K.learning_phase()], grad)
    return np.array(func([np.expand_dims(x, 0), learning_phase]))

##########################################################################################################
def get_gradients_at_layers(model, layer_index_list, no_of_nodes, image):
    '''
    The function get a list of layers, samples a number of nodes from each of the layers and returns the gradients of these nodes with respect to the input layer of the model evaluated on image.
    Input:
        model: the model on which the gradients are evaluated - an instance of tensorflow.keras.models.Model
        layer_index_list: a list of integers - specifies the indices of layers.
        no_of_nodes: a positive integer - specifies the number of nodes to be sampled from each of the layers.
        image: the input image on which the gradients are to be evaluated.
    Output: 
        a numpy array of shape (no_of_layers, no_of_nodes, shape_of_inputs_to_the_model) that contains the calculated gradients.
    '''

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

