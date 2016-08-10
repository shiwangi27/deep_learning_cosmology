#This is the py file! 
import numpy as np 
import os
import h5py
import sys
sys.path.insert(0,'/global/common/cori/software/theano/0.8.2/lib/python2.7/site-packages/')
import theano
import theano.tensor as T
sys.path.insert(0,'/global/common/cori/software/keras/1.02/lib/python2.7/site-packages/')
import keras

import numpy as np
from matplotlib import image as img

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.regularizers import l1
from keras import optimizers
#from keras.utils.visualize_util import plot

import matplotlib.pyplot as plt

def load_data():

    # Load the dataset
    
    # Here we give different data sets for the autoencoders
    dataurl = '/global/homes/s/ssingh79/data/'
    #hdf5file = 'conv_z02.h5'
    hdf5file = 'train_data_64k.h5'
    filepath = os.path.join(dataurl, hdf5file)
    
    print("Calling ", hdf5file, "......")
    # Call the load_data method to get back the Final training set. 
    dataset = filepath
    
    sample_size = 64000
    
    with h5py.File(dataset,'r') as hf:
        #train_set = hf['X_train'][0:1000,0:65536]
        train_set = hf['X_train'][:,:]
        print("Printing Train set ", train_set)
        print("X_train shape ", train_set.shape)
   
    #Create Training set and Validation set: 80 : 20 Randomly Sampling the images. 
    X = np.random.choice(1000, 1000, replace=False)
    split_percent = 0.80  
    
    '''
    print(X)
    #Get the random indices of images.  
    train_split = sample_size*split_percent
    train_index = X[0:train_split]
    valid_index = X[train_split:sample_size]
    
    train_x = train_set[train_index[:], : ]
    print("Training Set : ", train_x)
    print(train_x.shape) 
    
    valid_x = train_set[valid_index[:], : ]
    print("Validation Set : ", valid_x)
    print(valid_x.shape)
    
    return train_x, valid_x
    '''
    return train_set

def build_ae():

    encoding_dim = 12000
    
    input_img = Input(shape=(16384,))
    encoded = Dense(encoding_dim, init='normal', W_regularizer=l1(0.01), bias=True, activation ='relu', activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
    decoded = Dense(16384, activation='relu')(encoded)

    #reconstruction mapping
    autoencoder = Model(input=input_img,output=decoded)
    
    print("Encoded :", encoded)
    print("Decoded :", decoded)
    print("Autoencoder : ", autoencoder)
    
    # Compile the model
    sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.6, nesterov=True)
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error')
    
    #encoder model 
    encoder = Model(input=input_img, output=encoded)

    #decoder model
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    
    #x_train, x_valid = load_data()
    x_train = load_data() 
    
    #with h5py.File('/global/homes/s/ssingh79/temp_train_data.h5', 'w') as hf:
    #    print("Creating x_train h5py !")
    #    hf.create_dataset('x_train', data = x_train)
    
    #print(x_train[0].reshape(128,128))
    print(x_train.shape)
    
    
    print("Start training...........")
    #Train autoencoder
    autoencoder.fit(x_train,x_train,
                   nb_epoch=50,
                   batch_size=256,
                   shuffle=True,
                   validation_split = 0.1
                   #validation_data = (x_valid, x_valid)
                   )
    
    
    #----------------------------------------------------------------------
    
    
    # Visulaizing weights in Keras: 
    print("Weights Learnt by the autoencoder : ", autoencoder.get_weights())
    
    A = autoencoder.get_weights()
    print("Weight Shapes : ", A.shape)
    
    
    # Learning curve: 
    #plot(autoencoder, to_file= 'model.png')
  
    
    encoded_imgs = encoder.predict(x_train)
    decoded_imgs = decoder.predict(encoded_imgs)
    
    print(x_train[0].reshape(128,128))
    print(decoded_imgs[0].reshape(128,128))
    
    print("Decoded images : ", decoded_imgs)
    print(decoded_imgs.shape)
    
    #with h5py.File('/global/homes/s/ssingh79/out_images.h5', 'w') as hf:
    #    print("Creating decoded images that is reconstruction!")
    #    hf.create_dataset('decoded_imgs', data = decoded_imgs) 

    # Save Input and Reconstructed Images. 
    #plt.imshow(x_train[0].reshape(128,128), interpolation = 'None')
    #plt.colorbar()
    #plt.savefig('/global/homes/s/ssingh79/Output_files/train_x.png')
    #plt.close()
    
    img.imsave('/global/homes/s/ssingh79/Output_files/train_x.png', x_train[0].reshape(128,128))
    
    #plt.imshow(decoded_imgs[0].reshape(128,128), interpolation = 'None')
    #plt.savefig('/global/homes/s/ssingh79/Output_files/reconstruct_x.png')
    #plt.colorbar()
    #plt.close()
    
    img.imsave('/global/homes/s/ssingh79/Output_files/reconstruct_x.png', decoded_imgs[0].reshape(128,128))
   
build_ae()

