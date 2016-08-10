#The code here preprocesses the Cosmology input data. 

import numpy as np 
import h5py
from sklearn import preprocessing as pp 
import matplotlib 
from matplotlib import pyplot as plt

def read_h5py_dataset():  
    with h5py.File('/global/homes/s/ssingh79/data/conv_z02.h5', 'r') as hf:
        #print(hf.keys())
        print("Reading conv_z02.h5 ...........")
        X_train = hf['X_train'][0:1000, 0:1048576]        
        print("h5py shape ", X_train.shape)
        
        print("Training Data: ", X_train) 
        return X_train

def std_scale(normalize_images):
    
    ###### Mean-subtraction 
    std_scale = pp.StandardScaler().fit(normalize_images)
    normalized_std_scale = std_scale.transform(normalize_images)
    print("Standard Scale: ", normalized_std_scale)
    
    return normalized_std_scale
    
def min_max_scale(normalize_images):    
    
    # Normalize the data between [0, 1]. This is called Feature Scaling
    min_max_scaler = pp.MinMaxScaler()
    normalized_zero_to_one = min_max_scaler.fit_transform(normalize_images)
    print("Normalize [0,1] : ", normalized_zero_to_one)
    
    return normalized_zero_to_one
    
def mean_diff_min_max(normalize_images):
    
    #Mean difference and between [0,1]
    
    normalized_std_scale = std_scale(normalize_images)
    
    min_max_scaler_1 = pp.MinMaxScaler()
    normalized_zero_to_one_1 = min_max_scaler_1.fit_transform(normalized_std_scale)
    print(" Standard scale + Normalize [0,1] : ", normalized_zero_to_one_1)
    
    return normalized_zero_to_one_1
    
def max_abs_scale(normalize_images):
    
    # Normalize the data between [-1, 1]. This is called Feature Scaling
    max_abs_scaler = pp.MaxAbsScaler()
    normalized_neg_pos_one = max_abs_scaler.fit_transform(normalize_images)
    print("Normalize [-1,1] : ",normalized_neg_pos_one)
    
    return normalized_neg_pos_one
    
def mean_diff_max_abs_scale(normalize_images):
    
    # Normalization with both mean subtraction and between [-1,1] 
    
    normalized_std_scale = std_scale(normalize_images)
    
    max_abs_scaler_1 = pp.MaxAbsScaler()
    normalized_neg_pos_one_1 = max_abs_scaler_1.fit_transform(normalized_std_scale)
    print("Standard Scale + Normalize [-1,1] : ",normalized_neg_pos_one_1)
    
    return normalized_neg_pos_one_1

def create_h5py_dataset(normalize_images):
        
    # Call various functions and create h5py data files from returned numpy   
    #data_std_scale = std_scale(normalize_images)    
    #data_min_max = min_max_scale(normalize_images)
    data_mean_diff_min = mean_diff_min_max(normalize_images)
    #data_max_abs = max_abs_scale(normalize_images)
    #data_mean_diff_abs = mean_diff_max_abs_scale(normalize_images)
    
    # Create h5py data files with 5 techniques 
    with h5py.File('/global/homes/s/ssingh79/data/normalized_data.h5','w') as hf:
        # X_train is the training set needed for unsupervised learning. 
       
        print("Creating h5py data files and saving to ./data/normalized_data.h5 .........")
        #hf.create_dataset('data_std_scale', data = data_std_scale[0:10,:])
        #hf.create_dataset('data_min_max', data = data_min_max[0:10,:])
        hf.create_dataset('data_mean_diff_min', data = data_mean_diff_min[:,:]) 
        #hf.create_dataset('data_max_abs', data = data_max_abs[0:10,:])
        #hf.create_dataset('data_mean_diff_abs', data = data_mean_diff_abs[:,:])
        
        
    
def normalize_data():
    # This normalizes the data between [-1, 1]
    # A two step process which includes calculating the mean difference and then
    # normalizing by known standard formulae, here MaxAbsScaler to get data 
    # between [-1, 1].
    
    normalize_images = read_h5py_dataset()
    
    # Get the Mean, Minimum and Maximum values from the dataset 
    print("Mean = ", np.mean(normalize_images))
    print("Min = ", np.min(normalize_images))
    print("Max = ", np.max(normalize_images))
    
    create_h5py_dataset(normalize_images)

def read_normalized_data():
    
    # Show the original image
    with h5py.File('/global/homes/s/ssingh79/data/conv_z02.h5', 'r') as hr:
        org = hr['X_train'][0,:]
        org = np.reshape(org, (1024,1024))
        
        #%matplotlib inline
        #plt.imshow(org, interpolation='None')
        #plt.colorbar()
    
    # Read the normalized hdf5 data here
    with h5py.File('/global/homes/s/ssingh79/data/normalized_data.h5','r') as hf:
        print(hf.keys())
        #image1 = hf['data_std_scale'][:,:]
        #image2 = hf['data_min_max'][:,:]
        image3 = hf['data_mean_diff_min'][:,:]
        #image4 = hf['data_max_abs'][:,:]
        #image5 = hf['data_mean_diff_abs'][:,:]
        
        #print("Standard Scaling i.e Zero mean and unit co-variance: ",image1)
        #print("Normalize between [0,1]", image2)
        print("Standard scaling and normalize between [0,1]", image3)
        #print("Normalize between [-1,1]", image4)
        #print("Standard scaling and normalize between [-1,1]", image5)
        print(image3.shape)
        print(image3)
        #print(np.max("Maxi value = ", image3))
        #print(np.min("Mini value = ",image3))
        
        # Here we reshape the matrix to get images back again to 1024 X 1024 from flattened array. 
        # If you just change the row index from 0 to n=10. You can see all 10 set of images. 
        #print_image1 = np.reshape(image1[0,:],(1024,1024))
        #print_image2 = np.reshape(image2[0,:],(1024,1024))
        print_image3 = np.reshape(image3[0,:],(1024,1024))
        #print_image4 = np.reshape(image4[0,:],(1024,1024)) 
        #print_image5 = np.reshape(image5[0,:],(1024,1024)) 
        
        #%matplotlib inline
        #plt.imshow(print_image1, interpolation='None')
        #plt.colorbar()
        '''
        %matplotlib inline
        plt.subplot(321)
        plt.imshow(print_image1, cmap='Greys_r')
        plt.subplot(322)
        plt.imshow(print_image2, cmap='Greys_r')
        plt.subplot(323)
        plt.imshow(print_image3, cmap='Greys_r')
        plt.subplot(324)
        plt.imshow(print_image4, cmap='Greys_r')
        plt.subplot(325)
        plt.imshow(print_image4, cmap='Greys_r')
        plt.colorbar()        
    '''
    
#Run command to preprocess the data.
#normalize_data()

#Run the command to read preprocessed data. Comment above. 
read_normalized_data()