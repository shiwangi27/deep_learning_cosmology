#### need to module load python/2.7-anaconda
from astropy.io import fits
from scipy import misc
import numpy as np
import os
import matplotlib 
from matplotlib import pyplot as plt
import pickle, gzip
import h5py

# Debbie Dirs 
dataurl_deb = "/global/homes/d/djbard/project_lsst/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798/"

# Dir urls on srcatch and local host for m-series z=2 
#dataurl_from_scratch_dir = "/global/cscratch1/sd/ssingh79/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798/"
#dataurl_from_local_dir = "/Users/ssingh79/Downloads/DeepLearningTutorials-master/data/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798/"
#dataurl_from_homes_dir = "/global/homes/s/ssingh79/data/"
#base_filename = "fico_0200z_conv_"


dataurl_from_scratch_dir = '/global/cscratch1/sd/ssingh79/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.850/'
base_filename = "si85_0200z_conv_"


ouptut_data_dir = ""

def create_png_from_fits():
    for i in range(1,1001):
        if i<10:
            num = "000"+str(i)
        elif i<100:
            num="00"+str(i)
        elif i<1000:
            num = "0"+str(i)
        elif i==1000:
            num = str(i)
        else:
            print("gah!") 
              
        filepath = os.path.join(dataurl_from_local_dir,base_filename + str(num) + ".fit")
        #print(filepath)
        hdulist = fits.open(filepath)

        image = hdulist[0].data
        
        # I'm using the max/min pixel values -0.102/1.6, so that I'm making a consistent greyscale across the cosmologies. This covers everything at z=2. 
        misc.toimage(image, cmin=-0.102, cmax=1.6).save("fico-gifs/fico_0200z_conv_"+num+".png", )

def create_numpy_from_fits(): 

    # Create a numpy array to store images (numpy ndarray)
    # Initialize space for the numpy ndarray 
    image_dim = 1024*1024
    sample_size = 1000
    #flatten_images = np.zeros((sample_size, image_dim), dtype= 'float')
    
    flatten_images = np.zeros((1,1024*1024))
    
    for i in range(1,1001):
        if i<10:
            num = "000"+str(i)
        elif i<100:
            num="00"+str(i)
        elif i<1000:
            num = "0"+str(i)
        elif i==1000:
            num = str(i)
        else:
            print("gah!") 
                  
        filepath = os.path.join(dataurl_from_scratch_dir, base_filename + num + ".fit")
        #print(filepath)
        hdulist = fits.open(filepath)

        image = hdulist[0].data
        # Check type of values
        #print(image)
        #print(image.shape)
        #print(image.dtype)
            
        reshaped_image = np.reshape(image,(1,1024*1024))
        #print(reshaped_image)
        
        #image is a temporary numpy array which is to copied in every iteration as a row in flatten_image
        #image has dimension 1024 X 1024 its a 2D array, You want to reshape it to just a row!
        # After which you need to append.
    
        flatten_images = np.vstack((flatten_images , reshaped_image))
        #print(flatten_images)
        print("exiting Iteration", i)
    
    #Delete the first row initialization with 0 done before. 
    flatten_images = np.delete(flatten_images,(0),axis=0)
    
    print(flatten_images)
    print(flatten_images.shape)
    
    #Find the mean, min and max values in the dataset. 
    print("Mean = ", np.mean(flatten_images))
    print("Min = ", np.min(flatten_images))
    print("Max = " , np.max(flatten_images))
    
    #Visulaize the very last image in the dataset. 
    #%matplotlib inline
    #plt.imshow(image,interpolation='None')
    #plt.colorbar()
    
    return flatten_images

def create_h5py_dataset():
    
    flattened_images = create_numpy_from_fits()
    
    # Create h5py data files 
    with h5py.File('/global/cscratch1/sd/ssingh79/data/si85_z02.h5','w') as hf:
        # X_train is the training set needed for unsupervised learning. 
       
        print("Creating h5py data files and saving to ./data/si85_z02.h5.......")
        hf.create_dataset('X_train', data = flattened_images) 
        
def read_h5py_dataset(): 
    
    with h5py.File('/global/homes/s/ssingh79/data/conv_z02.h5', 'r') as hf:
        #print(hf.keys())
        print("Reading conv_z02.h5 ...........")
        X_train = hf['X_train'][0:1000, 0:1048576]
        print("h5py shape ", X_train.shape)
        
        print("Training Data: ", X_train)
    
# Oopsie Pickle needs some help with dataset larger than 1M floats, meh! 
# h5py comes to the rescue!
    
def pickle_dataset(): 

    flattened_images = create_numpy_from_fits()
    print("Flattened Images: ", flattened_images)
    
    #Pickle dataset with HIGHEST_PROTOCOL
    print("Pickling Dataset...")
    with open("/global/homes/s/ssingh79/data/conv_02z_data.pkl","wb") as f_pkl:
        pickle.dump(flattened_images, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done pickling!")
    
    #with gzip.open("/global/homes/s/ssingh79/data/conv_02z_data.pkl.gz","wb") as f_pklgz:
    #    pickle.dump(flattened_images, f_pklgz, protocol=pickle.HIGHEST_PROTOCOL)
    
def unpickle_dataset():
    #Load dataset from pickle files
    with open("/global/homes/s/ssingh79/data/conv_02z_data.pkl","rb") as f:
        print("Loading unpickled data...")
        train_set = pickle.load(f, encoding="UTF-8")
        print(train_set)
        
        unpickled_image = np.reshape(train_set[0],(1024,1024))
        print("Reshaped image", unpickled_image)
        #%matplotlib inline
        #plt.imshow(unpickled_image, interpolation='None')
        #plt.colorbar()
    
# To get Images in png format Uncomment this:
#create_png_from_fits()

# Run command to get a pickle gzip dataset desired for Training. 
#pickle_dataset()
#unpickle_dataset()

#Run command to get h5py files.
create_h5py_dataset()
#read_h5py_dataset()
