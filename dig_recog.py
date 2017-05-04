import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import eigh

import mlp

import time


'''
    Auxiliary function that plots every example on the dataset with their respective labels
        Parameters:
            data_set : (M x 28*28) matrix containing M rows of images
            norm_labels : a M-long vector containing the data_set labels
        Returns:
            -
'''
def plot(data_set, norm_labels):
    for i, p in enumerate(data_set[:, 1:]):
        plt.imshow(p.reshape(28,28), cmap='gray')
        print(p.reshape(28,28))
        plt.title(data_set[i,0])
        plt.title(norm_labels[i])
        plt.show()
        time.sleep(10)


'''
    Auxiliary function that normalizing the input transforming from grayscale (0-255) to MLP's input (0-1) and
    adjustment of output to be a vector insted of a digit
        Parameters:
            data_set : (M x F) matrix containing M rows of images and F columns of features
            n_rows : number of examples from the data_set that will be used
            train_portion : fraction of the cut data_set that will be used for training (0-1)
            validation :    boolean variable that indicates if the execution is in validation phase (benchmark purposes)
                            or production. In the second case we have no cut of the data_set for the training set, for
                            there are unique examples destined to testing.
        Returns:
            norm_train_images : normalized train images
            norm_train_images : normalized train labels
'''
def normalized_mnist_train_data(data_set, n_rows = 500, train_portion = 0.7, validation=True):
    if validation == False:
        # Input for every train example
        train_set = data_set[:, 1:]

        # Label for every train example
        train_labels = data_set[:, :1].astype(np.int64)


    else:
        train_idx = math.floor(train_portion * n_rows)

        # Input for every train example
        train_set = data_set[:train_idx, 1:]

        # Label for every train example
        train_labels = data_set[:train_idx, :1].astype(np.int64)

    # Normalization of the
    norm_train_images = np.divide(train_set, 255)
    norm_train_labels = [digit_to_vector(digit, 9) for digit in train_labels]

    return norm_train_images, norm_train_labels

'''
    Auxiliary function that normalizing the input transforming from grayscale (0-255) to MLP's input (0-1)
        Parameters:
            data_set : (M x F) matrix containing M rows of images and F columns of features
            n_rows : number of examples from the data_set that will be used
            train_portion : fraction of the cut data_set that will be used for training (0-1)
            validation :    boolean variable that indicates if the execution is in validation phase (benchmark purposes)
                            or production. In the second case we have no labels and there's no train portion on the
                            data_set.
        Returns:
            norm_train_images : normalized train images
            norm_train_images : normalized train labels
'''
def normalized_mnist_test_data(data_set, n_rows = 500, train_portion = 0.7, validation=True):
    # We don't have labels or train/test portions on production execution
    if validation == False:
        # Input for every test example
        test_set = data_set

        # Label for every test example
        norm_test_labels = None

    else:
        # Calculating test variables
        train_idx = math.floor(train_portion * n_rows)
        test_idx = math.floor((1-train_portion)*n_rows)

        # Input for every test example
        test_set = data_set[train_idx:train_idx+test_idx,1:]

        # Label for every test example
        test_labels = data_set[train_idx:train_idx+test_idx, :1].astype(np.int64)
        norm_test_labels = [digit_to_vector(digit, 9) for digit in test_labels]

    norm_test_images = np.divide(test_set, 255)

    return norm_test_images, norm_test_labels


'''
    Performs a test on a given trained model given a set of images and labels
        Parameters:
            model : trained MLP model
            norm_test_images : (M x F) matrix containing M rows of images and F columns of features
            norm_test_labels : (M x 1) matrix containing M rows of labels
        Returns:
            obtained : array of all obtained classes
            accuracy : percentage of classes that the model predicted correctly
'''
def mnist_test(model, norm_test_images, norm_test_labels):
    count = 0
    obtained = []
    for i, p in enumerate(norm_test_images):
        x_p = p[:]
        y_p = norm_test_labels[i]

        expected_class = vector_to_digit(y_p)
        obtained_class = vector_to_digit(model.classify(x_p))
        obtained.append(obtained_class)

        if expected_class == obtained_class:
            count += 1

    accuracy = count/len(norm_test_images)

    return obtained, accuracy


'''
    Auxiliary function that casts a digit (0-9) to a binary array[10]
        Parameters:
            n : number that's being converted
            d : numeric basis
        Returns:
            parsed_digit : d+1-dimensional array where parsed_digit[n] = 1
'''
def digit_to_vector(n, d):
    parsed_digit = np.zeros(d+1)
    parsed_digit[n] = 1
    return parsed_digit


'''
    Auxiliary function that casts a binary array[10] to a digit (0-9)
        Parameters:
            n : number that's being converted
        Returns:
            digit : index i where n[i] = 1

'''
def vector_to_digit(n):
    digit = np.argmax(n)
    return digit


'''
    Performs PCA on given data
        Parameters:
            data : n-by-p matrix where the rows correspond to observations and columns correspond to variables
            n_components : number of desired principal components
         Returns:
            w : sorted eigenvalues
            v : sorted eigenvalues
            idx : indexes of the sorted data

'''
def pca(data):
    # Rescale the data
    mi = data.mean(axis=0)
    data -= mi
    # data /= data.std(axis=0)

    cov_mat = np.cov(data, rowvar=False)
    w, v = eigh(cov_mat)

    # Sorting eigenvalues on descending order
    idx = np.argsort(w)[::-1]

    # Sort eigenvectors and eigenvalues according to the same index
    v = v[:,idx]
    w = w[idx]

    # Projection of the data to the new space
    rescaled_data = np.dot(data, v)

    return rescaled_data, w, v


'''
    Performs a dimensionality reduction to n_components on given data
        Parameters:
            data : n-by-p matrix where the rows correspond to observations and columns correspond to variables
            n_components : number of desired principal components
         Returns:
            pc : the new data_set containing only the principal components of the given observations
'''
def dimensionality_reduce(data, n_components = 100, feat_v = None):
    if feat_v is None:
        rescaled_data, _, v = pca(data)
        pc = rescaled_data[:, :n_components]

    else:
        mi = data.mean(axis=0)
        data -= mi

        v = feat_v
        pc = np.dot(data, v)[:, :n_components]

    return pc, v


'''
    Main function that's called to classify the MNIST dataset
        Parameters:
            n_components: number of principal components that we desire to use
            hidden_length : number of neurons on the hidden layer
            eta : size of the step given in the gradient direction
            threshold : minimum error acceptable
            n_rows : cut of the data_set rows that we desire to use
        Returns:
            accuracy : percentage of classes that the model predicted correctly
'''
def mnist_recognize(n_components, hidden_length = 10, eta = 0.01, threshold = 2e-2, n_rows = 500):
    # Reading train set
    data_set = pd.read_csv('train.csv', sep=',', header=0, dtype=np.float64).values

    # Normalizing images to grayscale and labels to vectors
    norm_train_images, norm_train_labels = normalized_mnist_train_data(data_set, n_rows = n_rows, train_portion=0.7,
                                                                       validation=False)

    # Calculating principal components
    norm_train_images, feat_v = dimensionality_reduce(data = norm_train_images, n_components =  n_components)

    # Showing images and labels
    # plot(data_set, train_idx, norm_train_labels)

    # Training neural network
    model = mlp.Model(i_neurons = n_components, h_neurons = hidden_length, o_neurons = 10)
    model.backpropagation(X = norm_train_images, Y = norm_train_labels, eta = eta, threshold = threshold)


    ''' Benchmark testing
    '''
    norm_test_images, norm_test_labels = normalized_mnist_test_data(data_set, n_rows = n_rows, train_portion=0.7)

    # Reducing dimensionality for the train set
    norm_test_images,_ = dimensionality_reduce(norm_test_images, n_components, feat_v)

    _, accuracy = mnist_test(model, norm_test_images, norm_test_labels)


    ''' Keggle submission testing
    '''
    test_set = pd.read_csv('test.csv', sep=',', header=0).values
    #norm_test_set = np.divide(test_set, 255)
    norm_test_images, _ = normalized_mnist_test_data(test_set, validation=False)
    norm_test_images, _ = dimensionality_reduce(norm_test_images, n_components, feat_v)

    for i, p in enumerate(norm_test_images):
        x_p = p[:]
        obtained_class = vector_to_digit(model.classify(x_p))

        print(i+1,',',obtained_class, sep='')

    return accuracy


def components_benchmark():
    nc = 70
    while nc > 0:
        print('N components: ', nc)
        start_time = time.time()
        accuracy = mnist_recognize(n_components= nc, hidden_length= 10, eta=0.01, n_rows = 3000, threshold=7e-2)
        end_time = time.time()
        duration = (end_time - start_time) / (60) # in minutes
        print('Taxa de acerto: ', accuracy)
        print('Duração: ', duration)
        nc -= 30

#components_benchmark()

start_time = time.time()
accuracy = mnist_recognize(n_components= 100, hidden_length= 10, eta=0.01, n_rows = 42000, threshold=3.5e-2)
end_time = time.time()
duration = (end_time - start_time) / (60) # in minutes
print('Taxa de acerto: ', accuracy)
print('Duração: ', duration)
