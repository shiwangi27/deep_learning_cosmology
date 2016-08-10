import numpy as np 
import os
import h5py
import sys
sys.path.insert(0,'/global/common/cori/software/theano/0.8.2/lib/python2.7/site-packages/')
import theano
import theano.tensor as T
sys.path.insert(0,'/global/common/cori/software/lasagne/0.1/lib/python2.7/site-packages/')

import lasagne
from lasagne import layers
from lasagne.layers import get_output, InputLayer, DenseLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
from lasagne.objectives import squared_error

sys.path.insert(0,'/global/common/cori/software/nolearn/0.6/')
import nolearn
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo
from nolearn.lasagne.base import TrainSplit
#from nolearn.lasagne.visualize import plot_loss
#from IPython.display import Image as IPImage
#from PIL import Image

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
    
    with h5py.File(dataset,'r') as hf: 
        #train_set = hf['X_train'][0:1000,0:65536] 
        train_set = hf['data_mean_diff_min'][0:1000,0:16384]
        print("Printing Train set ", train_set) 
        print("X_train shape ", train_set.shape[0])
        
    return train_set
        
def build_autoencoder(input_var=None):
    
    print('... loading data')
    X_train = load_data()
    
    #l_in = lasagne.layers.InputLayer((1000,128*128),)
    #l_hidden = lasagne.layers.DenseLayer(l_in,num_units=8000, nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotUniform())
    #l_out = lasagne.layers.DenseLayer(l_hidden,num_units=128*128)
    
    print("Start Training...........")
    
    ae = NeuralNet(layers = [
            (InputLayer, {'shape':(None, 128*128)}),
            (DenseLayer, {'num_units': 8000, 'W': lasagne.init.GlorotUniform(), 'b':lasagne.init.Constant(0.), 'nonlinearity':lasagne.nonlinearities.tanh }),
            (DenseLayer, {'num_units': 128*128, 'nonlinearity':lasagne.nonlinearities.tanh}),
            ],
            #input_shape = (None, 128*128),
            #hidden_num_units=8000,  # number of units in 'hidden' layer
            #output_nonlinearity=lasagne.nonlinearities.tanh,
            #output_num_units=128*128,  # 10 target values for the digits 0, 1, 2, ..., 9

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,
            
            train_split=TrainSplit(0.1),
            #objective=squared_error,
            #loss = None,
            objective_loss_function = squared_error,
                   
            max_epochs=50,
            verbose=1,           
        )
    
    ae.fit(X_train,X_train)
    
    print(ae)
    
    # Save the model ae in hdf5 file.
    
    
    # Plot the learning curve.
    #plot_loss(ae)
        
def main():
    
    
    input_var = T.tensor4('inputs')
    network = build_autoencoder(input_var)
    
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, input_var)
    
    loss = loss.mean()
    
    params = lasagne.layers.get_all_params(network, trainable= True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)
    
    train_fn = theano.function([input_var],loss, updates=updates)
    
    
build_autoencoder()
    


