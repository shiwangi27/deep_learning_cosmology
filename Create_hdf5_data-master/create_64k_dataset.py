# This is the script to create full 64,000 image dataset. Merges four .h5 files containing 16k samples each and then stacks them to create a huge numpy array with 64k samples! 

import numpy as np
import h5py
import os 

def read_batches(batch_num):
        
    dataURL = '/global/cscratch1/sd/ssingh79/si85_data'
    filename = 'segmented16k_data_'
    
    datapath = os.path.join( dataURL, filename + str(batch_num) + '.h5')
    print("The batch path is ", datapath)
    
    with h5py.File(datapath,'r') as hf: 
        # Read each 16k sample 
        batch = hf['X_train64k'][:,:]
    
    return batch 
    
def merge_data():
    
    size = 128 
    
    #Initialize batch images
    batch_64k = np.zeros((1,size*size))
    
    for i in range(1,5):
        # Call merge function for each batches.
        batch_16k = read_batches(i)
        
        # Stack up the batches returned.  
        batch_64k = np.vstack((batch_64k, batch_16k))
        
        print("Exiting iteration -------------", i)
        
    #Delete the first row initialization with 0 done before. 
    batch_64k = np.delete(batch_64k,(0),axis=0)
    
    print("Printing the full 64k sample dataset : ", batch_64k)
    print(batch_64k.shape)
    
    return batch_64k

def save_64k_data():
    
    X_train = merge_data()
    
    with h5py.File('/global/cscratch1/sd/ssingh79/si85_data/si85_train_data_64k.h5', 'w') as hf:
        # X_train is the training set needed for unsupervised learning. 
       
        print("Creating h5py data files and saving to ./si85_data/si85_train_data_64k.h5 .........")
        hf.create_dataset('X_train', data = X_train[:,:]) 
        
        
save_64k_data()        



