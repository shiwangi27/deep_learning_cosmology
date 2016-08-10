'''
Load Shapes, Compute basis
These basis be sparse
author: Mayur Mudigonda, March 2, 2015
'''

import numpy as np
import scipy.io as scio
#from scipy.misc import imread
import glob
from scipy.optimize import minimize 
import os
import sys
#import ipdb
import sparse_code_gpu
import time
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib
#import tables
import random
import utilities
import h5py
from visualize import *
from sklearn import preprocessing

outputURL = '/global/homes/s/ssingh79/scene-sparse-master/'

def adjust_LR(LR, iterations):
    T = 2000
    scale = 1.0/(1.0 + (iterations/T))
    new_LR = scale* LR
    print('.......................New Learning Rate is........................',new_LR)
    return new_LR

#Plot Reconstruction Error
def plot_residuals(residuals):
    plt.plot(residuals[1:])
    plt.xlabel('iterations')
    plt.ylabel('Residual of I - \phi a')
    plt.savefig('Reconstruction_Error.png')
    plt.close()
    return
#Plot SnR
def plot_SNR(SNR):
    plt.plot(SNR[1:])
    plt.xlabel('iterations')
    plt.ylabel('(SNR) ')
    plt.savefig('SNR.png')
    plt.close()
    return

def visualize_data(data,iteration,patchdim,image_shape=None):
    #Use function we wrote previously
    out_image = utilities.tile_raster_images(data.T,patchdim,image_shape,tile_spacing=(2,2))
    plt.imshow(out_image)
    plt.colorbar()
    savepath_image= 'vis_data'+ '_iterations_' + str(iteration) + '.png'
    plt.savefig(savepath_image)
    plt.close()
    return

def main(training_iter = 300, lam = 0.1 , basis_no = 50 ):
     #Environment Variables
    #DATA = os.getenv('DATA')
    #proj_path = DATA + 'scene-sparse-master/'
    write_path = outputURL + 'experiments/' + 'Run_one/'
    #data_dict = scio.loadmat(proj_path+'IMAGES.mat')
    #IMAGES = data_dict['IMAGES']
    #(imsize, imsize,num_images) = np.shape(IMAGES)
    #print('Could not get file handle. Aborting')
    
    # Here we give different data sets for the autoencoders
    dataurl = '/global/homes/s/ssingh79/data/'
    #hdf5file = 'conv_z02.h5'
    hdf5file = 'train_data_64k.h5'
    filepath = os.path.join(dataurl, hdf5file)
    
    print("Calling ", hdf5file, "......")
    # Call the load_data method to get back the Final training set. 
    sample_size = 64000
    dataset = filepath
    
    with h5py.File(dataset,'r') as hf:
        #train_set = hf['X_train'][0:1000,0:65536]
        train_set = hf['X_train'][:,:]
        print("Printing Train set ", train_set)
        print("X_train shape ", train_set.shape)
    
    ###### mean calculation train_set shape = (sample_size, 128*128) 
    # Pixel value across all samples mean. 
    #train_x_mean = train_set.mean(axis=0) 
    # Subtract mean value from the data. 
    #train_set = train_set - train_x_mean
    
    scaler = preprocessing.StandardScaler().fit(train_set)
    train_set = scaler.transform(train_set)
    
    #min_max_scaler = preprocessing.MinMaxScaler()
    #train_set = min_max_scaler.fit_transform(train_set)
    
    print("Mean : ", np.mean(train_set))
    print("Min : ", np.min(train_set))
    print("Max : ", np.max(train_set))
    
    shp = train_set.shape
    # Reshape the data
    data = train_set.transpose()
    #data = train_set.reshape(128,128,-1) 
    print("Data Tranform ... ") 
    
    #Inference Variables
    LR = 1e-1 
    #training_iter = 300 
    err_eps = 1e-3 
    orig_patchdim = 128 
    patch_dim = orig_patchdim**2 
    patchdim = np.asarray([0,0]) 
    patchdim[0] = orig_patchdim 
    patchdim[1] = orig_patchdim 
    sz = np.sqrt(patch_dim) 
    print('patchdim is ---',patchdim) 
    batch = 200 
    #data = np.zeros((orig_patchdim**2,batch)) 
    #basis_no = 1*(orig_patchdim**2) 
    
    border = 4 
    matfile_write_path = write_path+'IMAGES_' + str(orig_patchdim) + 'x' + str(orig_patchdim) + '__LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis' 

    #Making and Changing directory
    try:
        print('Trying to see if directory exists already')
        os.stat(matfile_write_path)
    except:
        print('Nope nope nope. Making it now')
        os.mkdir(matfile_write_path)

    try:
        print('Navigating to said directory for data dumps')
        os.chdir(matfile_write_path)
    except:
        print('Unable to navigate to the folder where we want to save data dumps')

    #Create object
    lbfgs_sc = sparse_code_gpu.LBFGS_SC(LR=LR,lam=lam,batch=batch,basis_no=basis_no,patchdim=patchdim,savepath=matfile_write_path)
    residual_list=[]
    sparsity_list=[]
    snr_list=[]
    for ii in np.arange(training_iter):
        tm1 = time.time()
        print('Loading new Data')
        #Moving the image choosing inside the loop, so we get more randomness in image choice
        
        print("DATA SHAPES : ", data.shape[0], data.shape[1])
        idx = np.random.randint(0,data.shape[1],batch)
        data_batch = data[...,idx] 
        lbfgs_sc.load_data(data_batch) 
        
        SNR_I_2 = np.var(data_batch)
        tm2 = time.time()
        #print('*****************Adjusting Learning Rate*******************')
        #adj_LR = adjust_LR(LR,ii)
        #lbfgs_sc.adjust_LR(adj_LR)
        print('Training iteration -- ',ii)
        #Note this way, each column is a data vector
        tm3 = time.time()
        prev_obj = 1e6 
        jj = 0
        ''' 
        while True: 
            obj,active_infer = lbfgs_sc.infer_coeff_gd()            
            if np.mod(jj,10)==0:
                print('Value of objective function from previous iteration of coeff update',obj)
            if (np.abs(prev_obj - obj) < err_eps) or (jj > 100):
                break
            else:
                prev_obj = obj
            jj = jj + 1
        '''
        active_infer,res = lbfgs_sc.infer_coeff()
        sparsity_list.append(active_infer)
        tm4 = time.time()
        residual,active,basis=lbfgs_sc.update_basis()
        residual_list.append(residual)
        tm5 = time.time()
        denom = lbfgs_sc.recon.get_value()
        denom_var = np.var(denom)
        snr = SNR_I_2/(denom_var)
        snr = 10*np.log10(snr)
        snr_list.append(snr)
        print('Time to load data in seconds', tm2-tm1)
        print('Infer coefficients cost in seconds', tm4-tm3)
        print('The value of active coefficients after we do inference ', active_infer)
        print('The value of residual after we do learning ....', residual)
        print('The SNR for the model is .........',snr)
        print('The value of active coefficients after we do learning ....',active)
        print('The mean norm of the basis is .....',np.mean(np.linalg.norm(basis,axis=0)))
        print('Updating basis cost in seconds',tm5-tm4)
        #residual_list.append(residual)
        if np.mod(ii,10)==0:
            print('Saving the basis now, for iteration ',ii)
            scene_basis = {
            'basis': lbfgs_sc.basis.get_value(),
            'residuals':residual_list,
            'sparsity':sparsity_list,
            'snr':snr_list
            }
            scio.savemat('basis',scene_basis)
            print('Saving basis visualizations now')
            lbfgs_sc.visualize_basis(ii,[10,7])
            print('Saving data visualizations now')
            visualize_data(data_batch,ii,patchdim,[10,10])
            # Call visualize.py to save reconstructed image data! 
            if np.mod(ii,20)==0:
                lbfgs_sc.visualize_basis_recon(data_batch, ii)
            print('Saving SNR')
            plot_SNR(snr_list)
            print('Saving R_error')
            plot_residuals(residual_list)
            print('Average Coefficients')
            lbfgs_sc.plot_mean_firing(ii)
            print('Visualizations done....back to work now')

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--epoch', type=int, dest='training_iter')
    parser.add_argument('--lambda', type=float, dest='lam')
    parser.add_argument('--basis', type=int, dest='basis_no')
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
    
    
   