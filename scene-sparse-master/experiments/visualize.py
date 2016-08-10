# Here's the code to visulaize the results

import os
import h5py
import numpy as np
import theano 
from theano import tensor as T

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def visualize_raw_data(data, iteration):
    # Save image as png to Output_files folder. 
    image = np.reshape(train_x[0,:],(128,128))
    train_fpath = os.path.join(outputURL, foldername + train_fname + ext)
    print("Saving Training image .............")
    plt.imsave(train_fpath, image)

    # Save histogram of pixels in an image. 
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    ax.hist((image), bins=20)
    ax.set_title("Raw image pixels Histogram Plot")
    ax.set_xlabel("Pixel values")
    ax.set_ylabel("Frequency")
    print("Saving histogram of raw image.....")
    tHist_fpath = os.path.join(outputURL, foldername + tHist_fname + ext)
    fig.savefig(tHist_fpath)
    plt.close(fig)
        
def visualize_normalized_data(norm_img, iteration): 
    # Visualize the normalized images. 
    print("Saving normalized training image ...........")
    
    for i in range(0,5):        
        savepath_image= 'normalized_im_' + str(i) + '_iterations_' + str(iteration) + '.png'
        norm_im = norm_img[:,i]
        vmin = norm_im.min()
        vmax = norm_im.max() 
        image = np.reshape(norm_im,(128,128))
        plt.figure()
        cbar = plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.colorbar(cbar)
        plt.savefig(savepath_image)
        plt.close()
    return
        
def visualize_reconstruction(recon_img, norm_img, iteration):
     # Reconstruction of images
    print("Saving SC reconstructed images .........") 
    
    for i in range(0,5):
        savepath_image= 'sc_reconstr_im_' + str(i) + '_iterations_' + str(iteration) + '.png'
        recon_im = recon_img[:,i]
        vmin = recon_im.min()
        vmax = recon_im.max()
        image = np.reshape(recon_im,(128,128))
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.savefig(savepath_image)
        plt.close()
        
        #### Visualize the Input images too with vmin and vmax values from reconstructed images! 
        visualize_normalized_data(norm_img, iteration)
        
    return








