import os 
import sys
import h5py 

import numpy as np
from sklearn import preprocessing

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

outputURL = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/'

def load_data(sample_size=1000, split_percent = 0.8, outfolder='Conv_ae_output'):

    # Load the dataset
    
    # Here we give different data sets for the autoencoders
    dataurl = '/global/homes/s/ssingh79/data/'
    #hdf5file = 'conv_z02.h5'
    #hdf5file = 'train_data_64k.h5'
    hdf5file = 'train_data_64k.h5'
    filepath = os.path.join(dataurl, hdf5file)
    
    outpath = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/' + outfolder
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    
    print("Calling ", hdf5file, "......")
    # Call the load_data method to get back the Final training set. 
    dataset = filepath
    
    with h5py.File(dataset,'r') as hf:
        #train_set = hf['X_train'][0:1000,0:65536]
        train_set_1 = hf['X_train'][:,:]
        print("Printing Train set ", train_set_1)
        print("X_train 1 shape ", train_set_1.shape)
        
    print("Calling si85_train_data_64k.h5 ......")    
    
    with h5py.File('/global/cscratch1/sd/ssingh79/si85_data/si85_train_data_64k.h5', 'r') as hf:
        train_set_2 = hf['X_train'][:,:]
        print("Printing Train set ", train_set_2)
        print("X_train 2 shape ", train_set_2.shape)
   

    #################### Preprocessing ###############################
    # We first pass the train_set to reshape_data(), then training data is split into training set and validation set & then both the data sets are normalized. The reshape funtion then reshapes the falttened images into 2D images of (128,128) which is our final dataset that goes as input to the Network! :)
    
    def train_valid_split(train_set_1, train_set_2, sample_size, split_percent):
        #Create Training set and Validation set: Randomly Sampling the images by %. 
        X1 = np.random.randint(0,train_set_1.shape[0], sample_size/2)
        X2 = np.random.randint(0,train_set_2.shape[0], sample_size/2)

        train_set_1 = train_set_1[X1[:], :]
        train_set_2 = train_set_2[X1[:], :]
        
        print('Theoretical model 1', train_set_1.shape) 
        print('Theoretical model 2', train_set_2.shape)
        
        # Create labels for model 1 and model 2
        train_y_1 = np.zeros((train_set_1.shape[0],1))
        train_y_2 = np.ones((train_set_2.shape[0],1)) 
        
        # Stacking the two models 
        train_data = np.vstack((train_set_1, train_set_2)) 
        label_data = np.vstack((train_y_1,train_y_2))
        
        train_set = np.hstack((train_data, label_data)) 
        
        X = np.random.randint(0,train_set.shape[0], sample_size)
        
        #Get the random indices of images.  
        train_split = int(sample_size*split_percent)
        valid_split = int(train_split + sample_size*((1 - split_percent)/2))
        train_index = X[0:train_split]
        valid_index = X[train_split:valid_split]
        test_index = X[valid_split:sample_size]
        
        # Training set
        train_x = train_set[train_index[:], :-1 ]
        train_y = train_set[train_index[:], -1:]
        train_y = train_y[:,0].astype('int32')
        #train_y = train_y.astype('int32')
        print("Training Set : ", train_x)
        print(train_x.shape, train_y.shape) 

        #Validation set
        valid_x = train_set[valid_index[:], :-1 ]
        valid_y = train_set[valid_index[:], -1:]
        valid_y = valid_y[:,0].astype('int32')
        print("Validation Set : ", valid_x)
        print(valid_x.shape, valid_y.shape)
        
        # Test set
        test_x = train_set[test_index[:], :-1 ]
        test_y = train_set[test_index[:], -1: ]
        test_y = test_y[:,0].astype('int32')
        print("Test Set : ", test_x)
        print(test_x.shape, test_y.shape)
        
        #visualize_raw_data(train_x)
        #visualize_raw_data(valid_x, outfolder)
        
        return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    def normalize(train_x, valid_x, test_x):
        # Normalization of dataset goes here : 
        
        scaler = preprocessing.StandardScaler().fit(train_x)
        print(scaler.mean_)
        print(scaler.var_)
        train_x = scaler.transform(train_x)
        valid_x = scaler.transform(valid_x)
        test_x = scaler.transform(test_x)
        
        print("Normalized Training data : ", train_x)
        print("Mean : ", np.mean( train_x))
        print("min : ", np.min(train_x))
        print("max : ", np.max(train_x))
        print("Valid set min : ", np.min(valid_x))
        print("Valid set max : ", np.max(valid_x))
        print("Test set min : ", np.min(test_x))
        print("Test set max : ", np.max(test_x))
        
        return train_x, valid_x, test_x 
    
    def reshape_data(train_set_1, train_set_2):
        # Call the train_valid_split creating random samples.
        
        train_x, train_y, valid_x, valid_y, test_x, test_y = train_valid_split(train_set_1 , train_set_2, sample_size, split_percent)
        
        # Normalize the training data and validation data. 
        train_x, valid_x, test_x = normalize(train_x, valid_x, test_x) 
        
        # Here goes the code to Reshape to 2D images. Return the 2D images as 
        # X_train and x_validation after reshaping. 
        train_x = train_x.reshape(-1,1,128,128)
        #print("After Reshaping, training data : ", train_x.shape)
        valid_x = valid_x.reshape(-1,1,128,128)
        #print("After Reshaping, validation data : ", valid_x.shape)
        test_x = test_x.reshape(-1,1,128,128)
        #print("After Reshaping, Testing data : ", test_x.shape)
        
        # Visualize the normlaized data. 
        #visualize_normalized_data(train_x, outfolder)
        
        return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    # Final dataset is here after all the preprocessing. 
    train_x, train_y, valid_x, valid_y, test_x, test_y = reshape_data(train_set_1, train_set_2)
    
    
    #print("Final Training data", train_x) 
    print(train_x.shape, train_y.shape)
    #print("Final Validation data", valid_x) 
    print(valid_x.shape, valid_y.shape)
    #print("Final Validation data", test_x) 
    print(test_x.shape, test_y.shape)
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y


