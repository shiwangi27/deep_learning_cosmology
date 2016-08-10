# This is a class we will use for Loading the Data for an Autoencoder. Since the training is unsupervised, 
# all we care about is the Training inputs and not training labels. 

from __future__ import print_function

__docformat__ = 'restructedtext en'

import h5py
import os
import sys
import timeit

import numpy as np 

import theano
import theano.tensor as T


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    '''
    #############
    # LOAD DATA #
    #############

    print('... loading data')

    # Load the dataset
    
    with h5py.File(dataset,'r') as hf: 
        #train_set = hf['X_train'][0:1000,0:65536] 
        train_set = hf['data_mean_diff_abs'][0:1000,0:16384]
        print("Printing Train set ", train_set) 
        print("X_train shape ", train_set.shape)
        
    # train_set format: tuple(input)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input. (Sorry no labels)

    def shared_dataset(data_x, borrow=True): 
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
    #   data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                               borrow=borrow)
        print("Printing shared_dataset ", shared_x)
    
        return shared_x 

    train_set_x = shared_dataset(train_set)
    print("Final Training Set ", train_set_x)
    return train_set_x 

#load_data('/global/homes/s/ssingh79/data/conv_z02.h5') 

