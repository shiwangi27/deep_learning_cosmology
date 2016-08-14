# Here's the code to visulaize the results

import os
import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE


# Building directory and file names. 
outputURL = '/global/homes/s/ssingh79/convolutional_autoencoder-master/output_files/'
train_fname = '/training_images'
norm_fname = '/normalized_images_'
tHist_fname = '/training_hist'
nHist_fname = '/norm_hist_'
rHist_fname = '/reconst_hist_'
reconst_fname = '/ae_reconstruct_'
lcurve_fname = '/learning_curve' 
vcurve_fname = '/valid_curve'
acurve_fname = '/accuracy_curve'
filters_fname = '/vis_filters'
tsne_fname = '/Tsne_plot'
sal_fname = '/sal_map'
active_fname = '/activations'
ext = '.png'

def visualize_raw_data(train_x, foldername):
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
        
def visualize_normalized_data(train_x, foldername):
    # Visualize the normalized images. 
    
    print("Saving histogram of normalized image.....") 
    for i in range(0,5):
        norm_fpath = os.path.join(outputURL, foldername + norm_fname + str(i) + ext)
        plt.figure()
        plt.imshow(train_x[i][0])
        plt.colorbar()
        plt.savefig(norm_fpath)
        plt.close()

        # Save histograms of pixels in the normalized image. 
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1,1,1) # one row, one column, first plot
        ax.hist((train_x[i][0]), bins=40)
        ax.set_title("Normalized image pixels Histogram Plot")
        ax.set_xlabel("Pixel values")
        ax.set_ylabel("Frequency")
        nHist_fpath = os.path.join(outputURL, foldername + nHist_fname + str(i) + ext)
        fig.savefig(nHist_fpath)
        plt.close(fig)
    
        
def visualize_learning_curve(cost_plot, valid_cost_plot, foldername):
    # Plot Learning curve.  
    print("Plotting the learning curve .........")
    plt.style.use('bmh')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    ax.plot(cost_plot)
    #ax.plot(valid_cost_plot)
    ax.set_title("Learning Curve")
    ax.set_ylabel("Training loss")
    ax.set_xlabel("Epochs")
    #ax.legend(['Training loss', 'Validation loss'], loc='upper right')
    lcurve_fpath = os.path.join(outputURL, foldername + lcurve_fname + ext)
    fig.savefig(lcurve_fpath)
    plt.close(fig)
    
def visualize_validation_curve(cost_plot, valid_cost_plot, foldername):
    # Plot Learning curve.  
    print("Plotting the valid learning curve .........")
    plt.style.use('bmh')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #ax.plot(cost_plot)
    ax.plot(valid_cost_plot)
    ax.set_title("Learning Curve")
    ax.set_ylabel("Validation loss")
    ax.set_xlabel("Epochs")
    #ax.legend(['Training loss', 'Validation loss'], loc='upper right')
    vcurve_fpath = os.path.join(outputURL, foldername + vcurve_fname + ext)
    fig.savefig(vcurve_fpath)
    plt.close(fig)    
    
def visualize_accuracy_curve(accuracy_plot, foldername):
    print("Plotting the accuracy curve .........")
    plt.style.use('bmh')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    ax.plot(accuracy_plot)
    ax.set_title("Accuracy Curve")
    ax.set_ylabel("Accuracy in %")
    ax.set_xlabel("Epochs")
    #ax.legend(['Training loss', 'Validation loss'], loc='upper right')
    acurve_fpath = os.path.join(outputURL, foldername + acurve_fname + ext)
    fig.savefig(acurve_fpath)
    plt.close(fig)    
    

def visualize_filters(foldername):
    print("Visualizing the learnt filters ............")
    model_fname = os.path.join(outputURL, foldername + '/model.npz')
    with np.load(model_fname) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    
    print(len(param_values))
    
    num_filtahz = len(param_values[0])
    n = int(np.floor(np.sqrt(num_filtahz)))
    
    plt.figure(figsize=(10, 10))
    plt.title('Layer activations')
    for i in range(n*n):
        plt.subplot(n,n, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(param_values[0][i],(5,5)), cmap='Greys_r')
        filters_fpath = os.path.join(outputURL, foldername + filters_fname + ext)
        plt.savefig(filters_fpath)
    plt.close()
    
def visualize_reconstruction(train_images, pred_images, foldername):
     # Reconstruction of images
    print("Saving reconstructed images .........") 
    print("Saving histogram of reconstructed image.....") 

    for i in range(0,5):
        reconst_fpath = os.path.join(outputURL, foldername + reconst_fname + str(i) + ext)
        # Visualize pairs of Training and Reconstructed Images!!
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.title('Normalized Image '+ str(i))
        plt.imshow(train_images[i][0])
        plt.colorbar()
        plt.subplot(2,2,2)
        plt.title('Reconstructed Image '+ str(i))
        plt.imshow(pred_images[i][0])
        plt.colorbar() 
        plt.savefig(reconst_fpath)
        plt.close()
        #plt.imsave(reconst_fpath, pred_images[i][0])
        
        # Save histograms of pixels in the normalized image. 
        #fig = plt.figure(figsize=(5, 4))
        #ax = fig.add_subplot(1,1,1) # one row, one column, first plot
        #ax.hist((pred_images[i][0]), bins=20)
        #ax.set_title("Reconstructed image pixels Histogram Plot")
        #ax.set_xlabel("Pixel values")
        #ax.set_ylabel("Frequency")
        #nHist_fpath = os.path.join(outputURL, foldername + rHist_fname + str(i) + ext)
        #fig.savefig(nHist_fpath) 
        #plt.close(fig)
        
def visualize_activations(active_images, foldername):
    print("Visualizing activations .........")
    active_fpath = os.path.join(outputURL, foldername + active_fname + ext)
    plt.figure(figsize=(10, 10))
    plt.title('Layer 1 activations')
    for i in range(1,26):
        plt.subplot(5, 5, i)
        plt.imshow(active_images[i][0], cmap='Greys_r')
        plt.axis('off')
        plt.savefig(active_fpath)
    plt.close()
        
def visualize_tsne(bn_vector, labels, foldername):
    model = TSNE(n_components=2, random_state=0, n_iter=2000)
    Tsne_plot = model.fit_transform(bn_vector)
    model_colors = labels #.reshape(labels.shape[0],1)
    plt.figure()
    c_list = [('r' if a == 0 else 'b') for a in model_colors]
    plt.scatter(Tsne_plot[:,0],Tsne_plot[:,1], color=c_list, s=100, alpha=.4)
    tsne_fpath = os.path.join(outputURL, foldername + tsne_fname + ext)
    plt.savefig(tsne_fpath)
    plt.close()
    return Tsne_plot 
    

def visualize_saliency_map(original_img, saliency, max_class, foldername):
    print("Visualizing Saliency map .........")
    saliency = saliency[0]
    max_class = max_class[0] 
    saliency = saliency[::-1].transpose(1, 2, 0)
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.subplot(2, 2, 1)
    plt.title('input')
    plt.imshow(original_img[0,:].reshape(128,128))
    plt.subplot(2, 2, 2)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    sal_fpath = os.path.join(outputURL, foldername + sal_fname + ext)
    plt.savefig(sal_fpath)
    plt.close()

    
    
    
    
    
    
    
    
    
            