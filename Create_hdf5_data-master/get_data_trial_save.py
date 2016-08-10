#### need to module load python/2.7-anaconda
from astropy.io import fits
from scipy import misc
import numpy as np
import os
import matplotlib 
from matplotlib import pyplot as plt

# Debbie Dirs 
dataurl_deb = "/global/homes/d/djbard/project_lsst/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798/"

# Dir urls on srcatch and local host for m-series z=2 
dataurl_from_scratch_dir = "/global/cscratch1/sd/ssingh79/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798/"
dataurl_from_local_dir = "/Users/ssingh79/Downloads/DeepLearningTutorials-master/data/m-series/m-512b240_Om0.260_Ol0.740_w-1.000_ns0.960_si0.798/"
base_filename = "fico_0200z_conv_"

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
                  
        filepath = os.path.join(dataurl_from_local_dir, base_filename + num + ".fit")
        #print(filepath)
        hdulist = fits.open(filepath)

        image = hdulist[0].data
        # Check type of values
        #print(image)
        #print(image.shape)
        #print(image.dtype)
            
        reshaped_image = np.reshape(image,(1,1024*1024))
        print(reshaped_image)
        
        #image is a temporary numpy array which is to copied in every iteration as a row in flatten_image
        #image has dimension 1024 X 1024 its a 2D array, You want to reshape it to just a row!
        # After which you need to append.
    
        flatten_images = np.vstack((flatten_images , reshaped_image))
        print(flatten_images)
    
    #print(flatten_images)
    print(flatten_images.shape)

    return flatten_images


def normalize_data():
    #This normalizes the data between [-1, 1]
    # A two step process which includes calculating the mean difference and then normalizing by 
    # known standard formulae. 

   


def store_pklgz_from_numpy(): 

    flattened_images = create_numpy_from_fits()


# To get Images in png format Uncomment this:
#create_png_from_fits()

# Run command to get a pickle gzip dataset desired for Training. 
store_pklgz_from_numpy()