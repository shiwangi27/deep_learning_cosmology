#Cosmology project
# This project is about applying Machine Learning more importantly Deep Learning Technqiues on #Cosmological data to examine the structure of the universe. The dataset to start off with is a 
# simulation of heat maps in the form of a 2D imaging data. 
# This is a wrapper for the implementation of Deep Learning Technqiues. 

import theano 
import os

from loadingdata import load_data 
#from dAE import test_dA 
#from dA import test_dA

# Denoising Autoencoders    

def call_loadingdata():
    # Here we give different data sets for the autoencoders
    dataurl = '/global/homes/s/ssingh79/data/'
    #hdf5file = 'conv_z02.h5'
    hdf5file = 'normalized_data.h5'
    filepath = os.path.join(dataurl, hdf5file)
    
    print("Calling ", hdf5file, "......")
    # Call the load_data method to get back the Final training set. 
    dataset = load_data(filepath)
    
    return dataset
        
def create_network():
    # Number of Visible nodes
    # Number of Hidden nodes
    # Number of Hidden Layers in case of a full network
   
    n_visible = 16384  
    n_hidden = 1000
    corruption_level = 0.3
    
    # Specified network parameters 
    return n_visible, n_hidden, corruption_level

def pass_parameters():
    # Load final dataset
    dataset = call_loadingdata()
    
    # Specify network parameters
    learning_rate = 0.05
    training_epochs = 20 
    batch_size = 16 
    output_folder = 'dA_plots'
    
    # Run dA class
    return (learning_rate,
            training_epochs,
            dataset,
            batch_size,
            output_folder)


#def test_preprocess():
    # Do preprocessing and create hdf5 file for training. Call get_data.py and preprocess.py 


#def test_display():
    #Display images from training 

#test_denoising_ae()
#pass_parameters()

