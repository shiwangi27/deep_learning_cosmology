# New .py for pure lasagne
# This is Convolutional Autoencoder using Theano and Lasagne wrapper. 

import os 
import sys
import h5py 
import time

import numpy as np
sys.path.insert(0,'/global/common/cori/software/theano/0.8.2/lib/python2.7/site-packages/')
import theano
import theano.tensor as T
sys.path.insert(0,'/global/common/cori/software/lasagne/0.1/lib/python2.7/site-packages/')
import lasagne

from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Conv2DLayer, Deconv2DLayer
from lasagne.layers import MaxPool2DLayer 
from lasagne.layers import ReshapeLayer 

from lasagne.objectives import squared_error

from theano.tensor.shared_randomstreams import RandomStreams

from visualize import *

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

outputURL = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/'

# ################## Load the dataset ################################
# Here the dataset is Cosmological maps with Image size 128x128 with total 
# sample size 64,000. The raw training data is unpacked from a h5py file 
# The data is then split into Training and Validation sets. Remember this
# is Unsupervised Learning so there are no Labels. 

def load_data(sample_size=100, split_percent = 0.8, outfolder='Conv_ae_output'):

    # Load the dataset
    
    # Here we give different data sets for the autoencoders
    dataurl = '/global/homes/s/ssingh79/data/'
    #hdf5file = 'conv_z02.h5'
    #hdf5file = 'train_data_64k.h5'
    hdf5file = 'train_samples_1M_p90.h5'
    filepath = os.path.join(dataurl, hdf5file)
    
    outpath = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/' + outfolder
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    
    print("Calling ", hdf5file, "......")
    # Call the load_data method to get back the Final training set. 
    dataset = filepath
    
    with h5py.File(dataset,'r') as hf:
        #train_set = hf['X_train'][0:1000,0:65536]
        train_set = hf['X_train'][:,:]
        print("Printing Train set ", train_set)
        print("X_train shape ", train_set.shape)
   

    #################### Preprocessing ###############################
    # We first pass the train_set to reshape_data(), then training data is split into training set and validation set & then both the data sets are normalized. The reshape funtion then reshapes the falttened images into 2D images of (128,128) which is our final dataset that goes as input to the Network! :)
    
    def train_valid_split(train_set, sample_size, split_percent):
        #Create Training set and Validation set: Randomly Sampling the images by %. 
        X = np.random.randint(0,train_set.shape[0], sample_size)
          
        #Get the random indices of images.  
        train_split = int(sample_size*split_percent)
        valid_split = int(train_split + sample_size*((1 - split_percent)/2))
        train_index = X[0:train_split]
        valid_index = X[train_split:valid_split]
        test_index = X[valid_split:sample_size]
        
        # Training set
        train_x = train_set[train_index[:], : ]
        print("Training Set : ", train_x)
        print(train_x.shape) 

        #Validation set
        valid_x = train_set[valid_index[:], : ]
        print("Validation Set : ", valid_x)
        print(valid_x.shape)
        
        # Test set
        test_x = train_set[test_index[:], : ]
        print("Test Set : ", test_x)
        print(test_x.shape)
        
        #visualize_raw_data(train_x)
        visualize_raw_data(valid_x, outfolder)
        
        return train_x, valid_x, test_x
    
    def normalize(train_x, valid_x, test_x):
        # Normalization of dataset goes here : 
        
        # Calculate Mean from the training set for subtraction: 
        train_x_mean = train_x.mean(axis=0)
        # Subtract mean values from train, valid and test data 
        train_x = train_x - train_x_mean
        valid_x = valid_x - train_x_mean
        test_x = test_x - train_x_mean
        
        # Divide the train, valid and test data with L2 norm! 
        X_norm = np.linalg.norm(train_x, axis=0)
        V_norm = np.linalg.norm(valid_x, axis=0)
        
        train_x = train_x/X_norm[np.newaxis,:]
        valid_x = valid_x/X_norm[np.newaxis,:]
        test_x = test_x/X_norm[np.newaxis,:]
        
        print("Normalized Training data : ", train_x)
        print("Mean : ", np.mean( train_x))
        print("min : ", np.min(train_x))
        print("max : ", np.max(train_x))
        print("Valid set min : ", np.min(valid_x))
        print("Valid set max : ", np.max(valid_x))
        print("Test set min : ", np.min(test_x))
        print("Test set max : ", np.max(test_x))
        
        return train_x, valid_x, test_x 
    
    def reshape_data(train_set):
        # Call the train_valid_split creating random samples. 
        train_x, valid_x, test_x = train_valid_split(train_set, sample_size, split_percent)
        
        # Normalize the training data and validation data. 
        train_x, valid_x, test_x = normalize(train_x, valid_x, test_x) 
        
        # Here goes the code to Reshape to 2D images. Return the 2D images as 
        # X_train and x_validation after reshaping. 
        train_x = train_x.reshape(-1,1,128,128)
        print("After Reshaping, training data : ", train_x.shape)
        valid_x = valid_x.reshape(-1,1,128,128)
        print("After Reshaping, validation data : ", valid_x.shape)
        test_x = test_x.reshape(-1,1,128,128)
        print("After Reshaping, Testing data : ", test_x.shape)
        
        # Visualize the normlaized data. 
        visualize_normalized_data(valid_x, outfolder)
        
        return train_x, valid_x, test_x
    
    # Final dataset is here after all the preprocessing. 
    train_x, valid_x, test_x = reshape_data(train_set)
    
    print("Final Training data", train_x) 
    print(train_x.shape)
    print("Final Validation data", valid_x) 
    print(valid_x.shape)
    print("Final Validation data", test_x) 
    print(test_x.shape)
    
    return train_x, valid_x, test_x  

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_conv_ae(input_var=None):
    
    # We will be building a Convolutional Autoencoder with multiple Conv2D and
    # Deconv2D layers. The Lasagne layers are packed step by step where output
    # of each layer is an input of the next layer. The function returns a 
    # network ready to be trained. 
    
    conv_num_filter = 64
    filter_size = 5
    corruption_p = 0.3 
    
    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 128 rows and 128 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 128, 128),
                                     input_var=input_var)
    
    print("Input shape : ", l_in.output_shape )
    
    #Formula : dim = {(image_size - filter_size + 2*padding)/stride} + 1 
    #Formula by convention : alpha =  {(i - k + 2*p)/s} + 1  

    # Add First Convolution2D layer -- (64, 64, 64)
    l_conv_first = lasagne.layers.Conv2DLayer(l_in, 
                                              num_filters=conv_num_filter, 
                                              filter_size=filter_size, 
                                              nonlinearity = lasagne.nonlinearities.rectify,
                                              stride=(2,2), 
                                              pad=2)
    
    print("Conv layer 1 : ", l_conv_first.output_shape)
    
    
    # Add the First Deconv layer -- (64, 128, 128)
    l_deconv_first = lasagne.layers.Deconv2DLayer(l_conv_first,
                                                num_filters= 1,
                                                filter_size = filter_size-1,
                                                nonlinearity = lasagne.nonlinearities.rectify,
                                                stride = (2,2),
                                                crop =1
                                                 )
    
    print("Deconv layer 1 : ", l_deconv_first.output_shape)

    
    print("Final layer Shape : ", l_deconv_first.output_shape)
    print("Final layer weights : ", l_deconv_first.W.get_value())
    
    print("Created Lasagne Layers & Network established !!! ")
    
    
    return l_deconv_first     
    
    
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, batchsize, shuffle=False):
    #assert len(inputs) == len(targets)    
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

        
# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

# batchszie, smapelsize, foldername, epochs, leraningrate

def main(num_epochs = 500, learning_rate=0.05, batch_size = 80, sample_size = 100, split_percent=0.8, outfolder = 'Conv_ae_output'):
    # Load the dataset
    print("Loading data...")
    X_train, X_valid, X_test = load_data(sample_size, split_percent, outfolder)
    
    # Prepare Theano variables for inputs
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    
    # Create neural network model 
    print("Building model and compiling functions.........")
    network = build_conv_ae(input_var)
    
    ######################### Training ##########################
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    
    print("Get the output layer : ", prediction)
    
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    
    # We could add some weight decay as well here, see lasagne.regularization.
    
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    print("Get all the parameters : ", params)
    
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.6)
    
    ###################### Validation/Testing #####################
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    
    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)
    
    
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    # Validation function
    val_fn = theano.function([input_var, target_var], test_loss)
    
    # Create a theano funtion to get the prediction. 
    predict_fn = theano.function([input_var], test_prediction)
    
    #*****You may have to write multiple theano function to get the Prediction per se
    
    # Finally, launch the training loop.
    print("Starting training...")
    # Create array to append cost. 
    cost_plot = []
    valid_cost_plot = []
    model_save_fname = outputURL + outfolder + '/model.npz'
    reconst_err_fname = outputURL + outfolder + '/reconst_error.npz'
    reconst_err_valid = outputURL + outfolder + '/reconst_error_valid.npz'
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #print(X_train.shape)
        #print(X_train.shape[0])
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            inputs = batch
            train_err += train_fn(inputs, inputs)  
            train_batches += 1
            
        # And a full pass over the validation data:
        #if (epoch%10==0):
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_batch_size = int(batch_size/(split_percent*10))
        for batch in iterate_minibatches(X_valid, val_batch_size, shuffle=False):
            inputs = batch
            err = val_fn(inputs, inputs)
            val_err += err
            #val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print("  validation accuracy:\t\t{:.2f} %".format(
        #    val_acc / val_batches * 100))
        
        # Append the mean cost in an array for plotting Learning curve. 
        cost_plot.append(train_err/train_batches)
        valid_cost_plot.append(val_err / val_batches)
        
        ############ Intermediate Plots and results #################
        # All the Visualization code is in visualize.py !
        
        if(np.mod(epoch,10)==0):
            # Call this function to plot the learning curve. 
            visualize_learning_curve(cost_plot, valid_cost_plot, outfolder)
            # Dump the network weights to a file like this:
            np.savez(model_save_fname, *lasagne.layers.get_all_param_values(network)) 
            print("Network Parameters saved!!!!!!!!")
            visualize_filters(outfolder)
            # Reconstruction of images
            visualize_reconstruction(pred_images, outfolder)
              
    print("Get all the Update Values : ", updates)
    
    # Save reconstruction error. 
    np.savez(reconst_err_fname, cost_plot) 
    np.savez(reconst_err_valid, valid_cost_plot)    
    
    print("Network Parameters W and b : ", lasagne.layers.get_all_param_values(network))
    
    ################### Visualization code ########################
    
    # Theano function predict gets the reconstructed output. 
    print("Running prediction function on Validation data")
    pred_images = predict_fn(X_valid) 
    
    # Save the Reconstruction images as hdf5 filessssss!!! 
    pred_images_fpath = outputURL + outfolder + '/reconstructed_maps.h5'
    with h5py.File(pred_images_fpath,'w') as hf:
        # X_train is the training set needed for unsupervised learning. 
        print("Creating hdf5 file for 100 pred images and saving to ./output_files/.......")
        hf.create_dataset('recon_maps', data = pred_images[0:100])
 
    
################# Run the code for Conv AE ####################
# Specify all the parameters that you want to optimize in the main function. 
# You can also use test.py to use this as a module and test with different parameters. 
# To run different parameters in parallel that is do hyperparameter optimization using multiple nodes on 
# the supercomputer, you can give the Command line arguments. Create batch scripts with parameters !!! 

if __name__ == '__main__': 
    # If you want to specify command line arguments. 
    if len(sys.argv) > 1:
        samples = int(sys.argv[1])
        batches = int(sys.argv[2])
        output = sys.argv[3]
        main(sample_size=samples, batch_size = batches, outfolder = output)      
    else:
        # Hyperparameter Optimizations here. 
        main(sample_size=60000, batch_size = 512, outfolder = 'Conv_ae_output_tvt')  
        