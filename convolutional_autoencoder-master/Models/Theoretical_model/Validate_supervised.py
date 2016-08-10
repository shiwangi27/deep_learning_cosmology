# New .py for pure lasagne
# This is Convolutional Autoencoder using Theano and Lasagne wrapper. 

import os 
import sys
import h5py 
import time

#from tqdm import tqdm
import numpy as np

import theano
import theano.tensor as T
sys.path.insert(0,'/global/common/cori/software/lasagne/0.1/lib/python2.7/site-packages/')
import lasagne

import lasagne.layers
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Conv2DLayer, Deconv2DLayer
from lasagne.layers import MaxPool2DLayer 
from lasagne.layers import ReshapeLayer 

from lasagne.objectives import categorical_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1

from theano.tensor.shared_randomstreams import RandomStreams

from visualize import *
from embed_tsne import *
from guided_backprop import * 

from load_theoretical_models import load_data

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

outputURL = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/'

# ################## Load the dataset ################################
# Here the dataset is Cosmological maps with Image size 128x128 with total 
# sample size 64,000. The raw training data is unpacked from a h5py file 
# The data is then split into Training and Validation sets. Remember this
# is Unsupervised Learning so there are no Labels. 

# see load_theoretical_models

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_conv_ae(input_var=None, use_dropout=False):
    
    # We will be building a Convolutional Autoencoder with multiple Conv2D and
    # Deconv2D layers. The Lasagne layers are packed step by step where output
    # of each layer is an input of the next layer. The function returns a 
    # network ready to be trained. 
    
    W_init =lasagne.init.HeNormal()
    conv_num_filter = 64
    conv_num_filter_1 = 128
    conv_num_filter_2 = 32
    
    dense_num_filter = 64 
    filter_size = 5
    hidden_units = 2048
    hidden_units_2 = 512 
    label_units = 2 
    
    corruption_p = 0.3 
    
    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 128 rows and 128 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    rng = np.random.RandomState(498)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    #do denoising here
    corrup_input = theano_rng.binomial(size=input_var.shape, n=1,
                                        p=1 - corruption_p,
                                        dtype=theano.config.floatX) * input_var
    
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 128, 128),
                                     input_var=corrup_input)
    
    print("Input shape : ", l_in.output_shape )
    
    if use_dropout:
        # Apply 20% dropout to the input data:
        l_in = lasagne.layers.DropoutLayer(l_in, p=0.2)
    
    #Formula : dim = {(image_size - filter_size + 2*padding)/stride} + 1 
    #Formula by convention : alpha =  {(i - k + 2*p)/s} + 1  

    # Add First Convolution2D layer -- (64, 64, 32)
    l_conv_first = lasagne.layers.Conv2DLayer(l_in, 
                                              num_filters=conv_num_filter_2, 
                                              filter_size=filter_size,
                                              nonlinearity = lasagne.nonlinearities.rectify,
                                              W = W_init,
                                              stride=(1,1), 
                                              pad=2)
    
    print("Conv layer 1 : ", l_conv_first.output_shape)
    
    
    l_max_pool_1 = lasagne.layers.MaxPool2DLayer(l_conv_first,
                                                 pool_size = (2,2))
    
    print("Max pool 1 : ", l_max_pool_1.output_shape)
    
    # Add Second Convolution2D Layer -- (32, 32, 64)
    l_conv_second = lasagne.layers.Conv2DLayer(l_max_pool_1, 
                                               num_filters=conv_num_filter_2, 
                                               filter_size=filter_size,
                                               nonlinearity = lasagne.nonlinearities.rectify,
                                               W = W_init,
                                               stride=(1,1),
                                               pad=2)
    
    print("Conv layer 2 : ", l_conv_second.output_shape)
    
    l_max_pool_2 = lasagne.layers.MaxPool2DLayer(l_conv_second,
                                                pool_size = (2,2))
    
    print("Max pool 2 : ", l_max_pool_2.output_shape)
    
    # Add Third Convolution2D Layer -- (16, 16, 64) 
    l_conv_third = lasagne.layers.Conv2DLayer(l_max_pool_2, 
                                               num_filters=conv_num_filter, 
                                               filter_size=filter_size,
                                               nonlinearity = lasagne.nonlinearities.rectify,
                                               W = W_init,
                                               stride=(1,1),
                                               pad=2)
    
    print("Conv layer 3 : ", l_conv_third.output_shape)
    
    l_max_pool_3 = lasagne.layers.MaxPool2DLayer(l_conv_third,
                                                pool_size = (2,2))
    
    print("Max pool 3 : ", l_max_pool_3.output_shape)
    
    
    # Add Fourth Convolution2D Layer -- (8, 8, 128)
    l_conv_fourth = lasagne.layers.Conv2DLayer(l_max_pool_3, 
                                               num_filters=conv_num_filter, 
                                               filter_size=filter_size,
                                               nonlinearity = lasagne.nonlinearities.rectify,
                                               W = W_init,
                                               stride=(1,1),
                                               pad=2)
    
    print("Conv layer 4 : ", l_conv_fourth.output_shape)
    
    l_max_pool_4 = lasagne.layers.MaxPool2DLayer(l_conv_fourth,
                                                pool_size = (2,2))
    
    print("Max pool 4 : ", l_max_pool_4.output_shape)
    
    # Add a fully-connected layer of ###### units, using the non-linear tanh, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    # Dense layer shape -- ( Vector of 1024, 2048 or 4096 ) 
    
    l_dense_1 = lasagne.layers.DenseLayer(l_max_pool_4, 
                                       num_units=hidden_units, 
                                       nonlinearity=lasagne.nonlinearities.rectify, 
                                       W=lasagne.init.GlorotUniform()) 
    
    print("Fully Connected layer: ", l_dense_1.output_shape)
    
    l_dense_2 = lasagne.layers.DenseLayer(l_dense_1, 
                                       num_units=hidden_units_2, 
                                       nonlinearity=lasagne.nonlinearities.rectify, 
                                       W=lasagne.init.GlorotUniform()) 
    
    #print("Fully Connected layer: ", l_dense_2.output_shape)
    
    l_softmax = lasagne.layers.DenseLayer(l_dense_1, 
                                       num_units=label_units, 
                                       nonlinearity=lasagne.nonlinearities.softmax, 
                                       W=lasagne.init.GlorotUniform()) 
    
    print("Softmax layer: ", l_softmax.output_shape) 
        
    print("Created Lasagne Layers & Network established !!! ")   
    
    return l_softmax, l_dense_1 
    
    
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

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)    
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

        
# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

# batchszie, smapelsize, foldername, epochs, leraningrate

def main(num_epochs = 300, learning_rate=0.05, batch_size = 80, sample_size = 100, split_percent=0.8, L2_lam = 0.001, use_dropout=False, outfolder = 'Conv_ae_output'):
    # Load the dataset
    print("Loading data...")
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_data(sample_size, split_percent, outfolder)
    
    print("Saving sampled training images.......")
    #save_training_images(X_train)
    
    # Prepare Theano variables for inputs
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets') 
    
    # Create neural network model 
    print("Building model and compiling functions.........")
    network, bottleneck_l = build_conv_ae(input_var)
    
    ######################### Training ##########################
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    
    print("Get the output layer : ", prediction)
    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    # We could add some weight decay as well here, see lasagne.regularization.
    #reg_layer = {bottleneck_l: L2_lam}
    #L2_reg = lasagne.regularization.regularize_layer_params_weighted(reg_layer, l2)
    # L2 Regularized loss:
    #loss = loss + L2_reg
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    print("Get all the parameters : ", params)
    
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9) # [0.5, 0.9, 0.95, 0.99] 
    
    ###################### Validation/Testing #####################
    
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    
    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)
    
    #################### Theano functions ######################
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    # Validation function
    val_fn = theano.function([input_var, target_var],[test_loss, test_acc])
    
    # Create a theano funtion to get the prediction. 
    predict_fn = theano.function([input_var], test_prediction)
    
    # T-SNE plots 
    hidden_prediction = lasagne.layers.get_output(bottleneck_l, deterministic=True)
    hidden_fn = theano.function([input_var], hidden_prediction)
    
    #Let's check the Saliency map!
    
    def compile_saliency_function(bottleneck_l):
        inp = input_var
        outp = lasagne.layers.get_output(bottleneck_l, deterministic=True)
        max_outp = T.max(outp,axis=1)
        saliency = theano.grad(max_outp.sum(), wrt=inp)
        max_class = T.argmax(outp, axis=1)
        return theano.function([inp], [saliency, max_class])

    #*****You may have to write multiple theano function to get the Prediction per se
    
    # Finally, launch the training loop.
    print("Starting training...")
    # Create array to append cost. 
    cost_plot = []
    valid_cost_plot = []
    accuracy_plot = []
    model_save_fname = outputURL + outfolder + '/model.npz'
    reconst_err_fname = outputURL + outfolder + '/reconst_error.npz'
    reconst_err_valid = outputURL + outfolder + '/reconst_error_valid.npz'
    bn_fname = outputURL + outfolder + '/bottleneck_layer.npz'
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #print(X_train.shape)
        #print(X_train.shape[0])
        for batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)  
            train_batches += 1
            
        # And a full pass over the validation data:
        #if (epoch%10==0):
        val_err = 0
        val_acc = 0
        val_batches = 0
        val_batch_size = X_valid.shape[0]
        for batch in iterate_minibatches(X_valid, Y_valid, val_batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        
        # Append the mean cost in an array for plotting Learning curve. 
        cost_plot.append(train_err/train_batches)
        valid_cost_plot.append(val_err/val_batches)
        accuracy_plot.append(val_acc / val_batches * 100)
        
        ############ Intermediate Plots and results #################
        # All the Visualization code is in visualize.py !
        
        if(np.mod(epoch,10)==0):
            print("Average Reconstruction Error -- ", np.mean(cost_plot))
            # Call this function to plot the learning curve. 
            visualize_learning_curve(cost_plot, valid_cost_plot, outfolder)
            visualize_validation_curve(cost_plot, valid_cost_plot, outfolder)
            visualize_accuracy_curve(accuracy_plot, outfolder)
            # Dump the network weights to a file like this:
            np.savez(model_save_fname, *lasagne.layers.get_all_param_values(network)) 
            print("Network Parameters saved!!!!!!!!")
            #visualize_filters(outfolder)
            # Theano function predict gets the reconstructed output. 
            print("Running prediction function on Validation data")
            #pred_images = predict_fn(X_train)
            # Reconstruction of images
            #visualize_reconstruction(X_train, pred_images, outfolder)
              
    #print("Get all the Update Values : ", updates)
    
    # Save reconstruction error. 
    np.savez(reconst_err_fname, cost_plot) 
    np.savez(reconst_err_valid, valid_cost_plot)    
    
    #print("Network Parameters W and b : ", lasagne.layers.get_all_param_values(network)) 
    #print("Bottleneck params # 3 ", lasagne.layers.get_output(network)) 
    
    ################### Visualization code ######################## 
    
    # Save the Reconstruction images as hdf5 filessssss!!! 
    #pred_images_fpath = outputURL + outfolder + '/reconstructed_maps.h5'
    #with h5py.File(pred_images_fpath,'w') as hf:
    #    # X_train is the training set needed for unsupervised learning. 
    #    print("Creating hdf5 file for 100 pred images and saving to ./output_files/.......")
    #    hf.create_dataset('recon_maps', data = pred_images[0:100])
    
    ######## SALIENCY MAPS ############
    
    # Using Guided Backprop compute the non-linearities again!
    relu = lasagne.nonlinearities.rectify
    relu_layers = [layer for layer in lasagne.layers.get_all_layers(network)
                   if getattr(layer, 'nonlinearity', None) is relu]
    modded_relu = GuidedBackprop(relu)  # important: only instantiate this once!
    for layer in relu_layers:
        layer.nonlinearity = modded_relu
    
    # Visualizing Saliency map! 
    saliency_fn = compile_saliency_function(bottleneck_l) 
    X_sal, max_class = saliency_fn(X_train)
    visualize_saliency_map(X_train, X_sal, max_class, outfolder)
    
    ######### Visualize TSNE ############ 
    print("Saving T-sne vector")
    bn_vector = hidden_fn(X_valid)
    # Save the hidden layer output:
    np.savez(bn_fname, bn_vector)
    print("Visualizing t-sne")
    print("BN VECTOR ------", bn_vector.shape)
    Tsne_vector = visualize_tsne(bn_vector, Y_valid, outfolder)
    # Embed Images into TSNE plot : 
    embed_img_into_tnse(X_train, Tsne_vector, outfolder)
    
    ## Save Saliency feature map! 
    sal_fpath  = outputURL + outfolder + '/saliency_maps.h5'
    with h5py.File(sal_fpath,'w') as hf:
        # X_train is the training set needed for unsupervised learning. 
        print("Creating hdf5 file for Salinecy maps saving to ./output_files/.......")
        hf.create_dataset('X_sal', data = X_sal[0:100])
        hf.create_dataset('max_class', data = max_class[0:100])
    
################# Run the code for Conv AE ####################
# Specify all the parameters that you want to optimize in the main function. 
# You can also use test.py to use this as a module and test with different parameters. 
# To run different parameters in parallel that is do hyperparameter optimization using multiple nodes on 
# the supercomputer, you can give the Command line arguments. Create batch scripts with parameters !!! 

if __name__ == '__main__': 
    # If you want to specify command line arguments. 
    # Hyperparameter Optimizations here. 
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='Command line options')
        parser.add_argument('--epochs', type=int, dest='num_epochs')
        parser.add_argument('--samples', type=int, dest='sample_size')
        parser.add_argument('--batches', type=int, dest='batch_size')
        parser.add_argument('--lr', type=float, dest='learning_rate')
        parser.add_argument('--output', type=str, dest='outfolder')
        parser.add_argument('--lam', type=float, dest='L2_lam')
        
        args = parser.parse_args(sys.argv[1:])
        main(**{k:v for (k,v) in vars(args).items() if v is not None})
    else:
        main(num_epochs = 300, sample_size=100, batch_size = 80, learning_rate=5e-4, use_dropout=False, L2_lam = 5e-4, outfolder = 'Valid_Supervised_CNN') 
        