import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import mlp

import time

def plot(data_set, train_idx, norm_train_labels):
    for i, p in enumerate(data_set[:train_idx,1:]):
        plt.imshow(p.reshape(28,28), cmap='gray')
        print(p.reshape(28,28))
        plt.title(data_set[i,0])
        plt.title(norm_train_labels[i])
        plt.show()
        time.sleep(10)

def normalized_mnist_train_data(data_set, n_rows = 500, train_portion = 0.7):
    train_idx = math.floor(train_portion * n_rows)

    # Input for every example
    train_set = data_set[:train_idx, 1:]
    norm_train_images = train_set / 255

    # Label for every example
    train_labels = data_set[:train_idx, :1]
    norm_train_labels = [digit_to_vector(digit, 9) for digit in train_labels]

    return norm_train_images, norm_train_labels


def normalized_mnist_test_data(data_set, n_rows = 500, train_portion = 0.7):
    train_idx = math.floor(train_portion * n_rows)

    # Calculating test variables
    test_idx = math.floor((1-train_portion)*n_rows)
    test_set = data_set[train_idx:train_idx+test_idx,1:]
    norm_test_images = np.divide(test_set, 255)
    test_labels = data_set[train_idx:train_idx+test_idx, :1]
    norm_test_labels = [digit_to_vector(digit, 9) for digit in test_labels]

    return norm_test_images, norm_test_labels


def mnist_test(model, norm_test_images, norm_test_labels):
    count = 0
    for i, p in enumerate(norm_test_images):
        x_p = p[:]
        y_p = norm_test_labels[i]

        expected_class = vector_to_digit(y_p)
        obtained_class = vector_to_digit(model.classify(x_p))
        if expected_class == obtained_class:
            count += 1

    accuracy = count/len(norm_test_images)

    return count, accuracy


# Auxiliary function that casts a digit (0-9) to a binary array[10]
def digit_to_vector(n, d):
    parsed_digit = np.zeros(d+1)
    parsed_digit[n] = 1
    return parsed_digit

def vector_to_digit(n):
    return np.argmax(n)


def mnist_recognize(hidden_length = 10, eta = 0.1, threshold = 0.05):
    # Reading train set
    data_set = pd.read_csv('train.csv', sep=',', header=0).values

    norm_train_images, norm_train_labels = normalized_mnist_train_data(data_set)

    # Showing images and labels
    # plot(data_set, train_idx, norm_train_labels)

    # Training neural network
    model = mlp.Model(28*28, hidden_length, 10)
    model.backpropagation(norm_train_images, norm_train_labels, eta, threshold)

    ''' Keggle submission testing
    Testing
    test_set = pd.read_csv('test.csv', sep=',', header=0).values
    norm_test_set = np.divide(test_set, 255)

    for p in norm_test_set:
        x_p = p[:]
        obtained_class = model.classify(x_p)
    '''

    norm_test_images, norm_test_labels = normalized_mnist_test_data(data_set)
    _, accuracy = mnist_test(model, norm_test_images, norm_test_labels)


    print('Taxa de acerto: ', accuracy)




start_time = time.time()
mnist_recognize()
end_time = time.time()
duration = (end_time - start_time) / (60) # in minutes
print(duration)
