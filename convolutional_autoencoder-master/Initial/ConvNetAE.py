# python version 
import os 
import pickle
import urllib
import sys
import h5py

sys.path.insert(0,'/global/common/cori/software/theano/0.8.2/lib/python2.7/site-packages/')
import theano
sys.path.insert(0,'/global/common/cori/software/lasagne/0.1/lib/python2.7/site-packages/')
import lasagne
sys.path.insert(0,'/global/common/cori/software/nolearn/0.6/')
import nolearn

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator, PrintLayerInfo
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import squared_error
from lasagne.nonlinearities import tanh, rectify

from shape import ReshapeLayer

from IPython.display import Image as IPImage
#from PIL import Image

# Importing modules for creating lasagne layers. 
from lasagne.layers import get_output, InputLayer, DenseLayer, Deconv2DLayer, Upscale2DLayer 
from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow
from lasagne.layers import Deconv2DLayer 
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')
    
#from nolearn.lasagne import TrainSplit

def load_data():

    # Load the dataset
    
    # Here we give different data sets for the autoencoders
    dataurl = '/global/homes/s/ssingh79/data/'
    #hdf5file = 'conv_z02.h5'
    hdf5file = 'segment128_data.h5'
    filepath = os.path.join(dataurl, hdf5file)
    
    print("Calling ", hdf5file, "......")
    # Call the load_data method to get back the Final training set. 
    dataset = filepath
    
    sample_size = 1000
    
    with h5py.File(dataset,'r') as hf:
        #train_set = hf['X_train'][0:1000,0:65536]
        train_set = hf['data_mean_diff_min'][0:1000,:]
        print("Printing Train set ", train_set)
        print("X_train shape ", train_set.shape)
   
    #Create Training set and Validation set: 80 : 20 Randomly Sampling the images. 
    X = np.random.choice(1000, 1000, replace=False)
    split_percent = 0.90  
    
    #print(X)
    #Get the random indices of images.  
    train_split = sample_size*split_percent
    train_index = X[0:train_split]
    valid_index = X[train_split:sample_size]
    
    train_x = train_set[train_index[:], : ]
    print("Training Set : ", train_x)
    print(train_x.shape) 
    
    valid_x = train_set[valid_index[:], : ]
    print("Validation Set : ", valid_x)
    print(valid_x.shape)
    
    return train_x, valid_x  

def build_conv_ae():
    
    X_train, X_valid = load_data()
    
    # reshape from (sample_size, 128*128) to 4D tensor (sample_size, 1, 128, 128)
    X_train = np.reshape(X_train, (-1, 1, 128, 128))
    print('X type and shape:', X_train.dtype, X_train.shape)
    print('X.min():', X_train.min())
    print('X.max():', X_train.max())
    
    # we need our target to be 1 dimensional
    X_out = X_train.reshape((X_train.shape[0], -1))
    print('X_out:', X_out.dtype, X_out.shape)
    
    conv_num_filters = 32
    filter_size = 5
    pool_size = 2
    encode_size = 32
    dense_mid_size = 4096
    pad_in = 'valid'
    pad_out = 'full'
    
    #Create Lasagne Layers!
    layers = [
        (InputLayer, {'shape': (None, X_train.shape[1], X_train.shape[2], X_train.shape[3])}), 
        (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
        (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
        (MaxPool2DLayerFast, {'pool_size': pool_size}),
        (Conv2DLayerFast, {'num_filters': 2*conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
        (MaxPool2DLayerFast, {'pool_size': pool_size}),
        (ReshapeLayer, {'shape': (([0], -1))}),
        (DenseLayer, {'num_units': dense_mid_size}),
        (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
        (DenseLayer, {'num_units': dense_mid_size}),
        (DenseLayer, {'num_units': 1600}),
        (ReshapeLayer, {'shape': (([0], 2*conv_num_filters, 5, 5))}),
        (Upscale2DLayer, {'scale_factor': pool_size}),
        (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
        (Upscale2DLayer, {'scale_factor': pool_size}),
        (Conv2DLayerSlow, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
        (Conv2DLayerSlow, {'num_filters': 1, 'filter_size': filter_size, 'pad': pad_out}),
        (ReshapeLayer, {'shape': (([0], -1))}),
    ] 
    
    #Create Network 
    ae = NeuralNet(
        layers=layers,
        max_epochs=50,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,

        
        objective_loss_function = squared_error,
        verbose=1
        
    )
    
    # Begin Training. 
    ae.fit(X_train, X_out)
    print 
    
    
build_conv_ae()

