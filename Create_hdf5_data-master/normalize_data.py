import numpy as np 
import h5py
#from sklearn import preprocessing as pp 
#from sklearn.preprocessing import normalize
from numpy import linalg
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import image as img 


def read_h5py_dataset():  
    with h5py.File('/global/homes/s/ssingh79/data/train_data_64k.h5', 'r') as hf:
        #print(hf.keys())
        print("Reading train_data_64k.h5 ...........")
        X_train = hf['X_train'][:, :]        
        print("h5py shape ", X_train.shape)
        
        print("Training Data: ", X_train)
    
    #%matplotlib inline
    #plt.imshow(np.reshape(X_train[0,:],(128,128)), interpolation='none')
    #plt.colorbar()
    #plt.savefig('/global/homes/s/ssingh79/Output_files/train_x1.png')
    #plt.close()
    
    img.imsave('/global/homes/s/ssingh79/Output_files/train_x1.png', np.reshape(X_train[0,:],(128,128)) )
    
    #Histogram
    #plt.hist(X_train[0,:], bins=20)
    #plt.title("Image[0] histogram of pixels")
    #plt.xlabel("Pixel value")
    #plt.ylabel("Frequency")
    #plt.plot
    #plt.savefig('/global/homes/s/ssingh79/Output_files/data_sample_hist.png')
    
    return X_train
    
def normalizer():
    X_train = read_h5py_dataset()
    X_norm = np.linalg.norm(X_train, axis=0)
    X_train = X_train/X_norm[np.newaxis,:]
    print("Normalized : ", X_train)
    print("Mean : ", np.mean( X_train))
    print("min : ", np.min(X_train))
    print("max : ", np.max(X_train))
    
    image = np.reshape(X_train[0,:],(128,128))
    
    #%matplotlib inline
    #plt.imshow(image, interpolation='none')
    #plt.colorbar()
    #plt.savefig('/global/homes/s/ssingh79/Output_files/normalized_x1.png')
    #plt.close()
    
    img.imsave('/global/homes/s/ssingh79/Output_files/normalized_x1.png', image)
    
    #Histogram
    plt.hist(X_train[0,:], bins=20)
    plt.title("Image[0] histogram of pixels")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.plot
    plt.savefig('hist_norm.png')
    
    
normalizer()
#def visualize_norm():