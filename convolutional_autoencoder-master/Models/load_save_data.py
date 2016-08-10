import os 
import sys
import h5py 
import time

import numpy as np
from sklearn import preprocessing

def load_data(sample_size=100, split_percent = 0.8, outfolder='Conv_ae_output'):

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
        #visualize_raw_data(valid_x, outfolder)
        
        return train_x, valid_x, test_x
    
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
        print("Variance", np.var(train_x))
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
        #print("After Reshaping, training data : ", train_x.shape)
        valid_x = valid_x.reshape(-1,1,128,128)
        #print("After Reshaping, validation data : ", valid_x.shape)
        test_x = test_x.reshape(-1,1,128,128)
        #print("After Reshaping, Testing data : ", test_x.shape)
        
        # Visualize the normlaized data. 
        #visualize_normalized_data(train_x, outfolder)
        
        return train_x, valid_x, test_x
    
    # Final dataset is here after all the preprocessing. 
    train_x, valid_x, test_x = reshape_data(train_set)
        
    #print("Final Training data", train_x) 
    print(train_x.shape)
    #print("Final Validation data", valid_x) 
    print(valid_x.shape)
    #print("Final Validation data", test_x) 
    print(test_x.shape)
    
    return train_x, valid_x, test_x 

def main():
    X_train, X_valid, X_test = load_data(sample_size = 10000, split_percent=0.8)
    
    print("Creating training, Validation and Test h5 files ....................")
    
    with h5py.File('/global/homes/s/ssingh79/data/train_caffe.h5','w') as hf:
        hf.create_dataset('data', data=X_train)
        
    with h5py.File('/global/homes/s/ssingh79/data/valid_caffe.h5','w') as hf:
        hf.create_dataset('data', data=X_valid)
        
    with h5py.File('/global/homes/s/ssingh79/data/test_caffe.h5','w') as hf:
        hf.create_dataset('data', data=X_test)

main()