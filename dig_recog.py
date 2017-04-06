import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import mlp

import time

def plot(x_p, y_p):
    for i in range(x_p.shape[0]):
        img=x_p[i]
        img=img.reshape((28,28))
        plt.imshow(img,cmap='gray')
        plt.title(y_p[i,0])
        plt.waitforbuttonpress()

def recognize(hidden_length = 4, eta = 0.1, threshold = 0.1, n_rows = 100, train_portion = 0.7):
    # Setting train set
    data_set = pd.read_csv('train.csv', sep=',', header=0).values

    train_idx = math.floor(train_portion * n_rows)

    train_set = data_set[:train_idx, 1:]
    norm_train_images = train_set / 255
    train_labels = data_set[:train_idx, :1]
    norm_train_labels = [digit_to_vector(digit, 9) for digit in train_labels]

    # for i, p in enumerate(data_set[:train_idx,1:]):
    #     plt.imshow(p.reshape(28,28), cmap='gray')
    #     # print(p.reshape(28,28))
    #     # plt.title(data_set[i,0])
    #     plt.title(norm_train_labels[i])
    #     plt.show()
    #     time.sleep(3)

    # Training neural network
    model = mlp.Model(28*28, hidden_length, 10)
    model.backpropagation(norm_train_images, norm_train_labels, threshold)

    # Testing
    # test_set = pd.read_csv('test.csv', sep=',', header=0).values
    # norm_test_set = np.divide(test_set, 255)
    #
    # for p in norm_test_set:
    #     x_p = p[:]
    #     obtained_class = model.classify(x_p)

    test_idx = math.floor((1-train_portion)*n_rows)
    test_set = data_set[train_idx:train_idx+test_idx,1:]
    norm_test_set = np.divide(test_set, 255)
    test_labels = data_set[train_idx:train_idx+test_idx, :1]
    norm_test_labels = [digit_to_vector(digit, 9) for digit in test_labels]

    count = 0
    for i, p in enumerate(norm_test_set):
        x_p = p[:]
        y_p = norm_test_labels[i]

        expected_class = vector_to_digit(y_p)
        obtained_class = vector_to_digit(model.classify(x_p))
        if expected_class == obtained_class:
            count += 1

    print('>', count/len(norm_test_set))

def digit_to_vector(n, d):
    parsed_digit = np.zeros(d+1)
    parsed_digit[n] = 1
    return parsed_digit

def vector_to_digit(n):
    return np.argmax(n)

recognize()
