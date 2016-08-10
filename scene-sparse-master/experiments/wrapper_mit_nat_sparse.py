'''
Load Shapes, Compute basis
These basis be sparse
author: Mayur Mudigonda, March 2, 2015
'''

import numpy as np
import scipy.io as scio
from scipy.misc import imread
import glob
from scipy.optimize import minimize 
import os
import ipdb
import sparse_code_gpu
import time
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib
import tables
import utilities
import h5py 

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
    out_image = utilities.tile_raster_images(data.T,patchdim,image_shape)
    plt.imshow(out_image,cmap=cm.Greys_r)
    savepath_image= 'vis_data'+ '_iterations_' + str(iteration) + '.png'
    plt.savefig(savepath_image)
    plt.close()
    return


def load_list(filename_no_extension):
    """
    :type filename_no_extension: A filename string without the .py extension
    """
    print('loading file ' + filename_no_extension)
    module = importlib.import_module(filename_no_extension)
    assert isinstance(module, object)
    assert isinstance(module.content, list)
    return module.content

def make_data(file_hndl,file_list,batch):

    data=np.zeros([32*32,batch])
    gen_idx = np.random.randint(0,len(file_list),batch)
    idx = 0
    for ii in np.arange(batch):
        try:
            data[:,idx] = file_hndl.root._f_get_child(file_list[gen_idx[ii]])[:]
        except:
            print('Could not extract indoor image with id ',gen_idx[ii])
        idx = idx + 1

    return data

if __name__ == "__main__":

    #Environment Variables
    DATA = os.getenv('DATA')
    proj_path = DATA + 'scene-sparse/' 
    write_path = proj_path + 'experiments/' 

    print('Loading File lists') 
    indoor_list=load_list('indoor_file_list') 
    outdoor_list=load_list('outdoor_file_list') 
    try: 
        h= tables.open_file('/media/mudigonda/Gondor/Data/scene-sparse/places32_whitened.h5','r') 
    except: 
        print('Could not get file handle. Aborting')
    #Inference Variables 
    LR = 1e-1 
    training_iter = 10000 
    lam = 1e-1 
    err_eps = 1e-3 
    orig_patchdim = 32 
    patchdim = np.asarray([0,0]) 
    patchdim[0] = orig_patchdim 
    patchdim[1] = orig_patchdim 
    print('patchdim is ---',patchdim) 
    batch = 200 
    basis_no =1*(orig_patchdim**2) 
    matfile_write_path = write_path+'outdoor_LR_'+str(LR)+'_batch_'+str(batch)+'_basis_no_'+str(basis_no)+'_lam_'+str(lam)+'_basis'

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
	data=make_data(h,outdoor_list,batch)
	lbfgs_sc.load_data(data)
	SNR_I_2 = np.var(data)
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
            lbfgs_sc.visualize_basis(ii,[orig_patchdim,orig_patchdim])
            print('Saving data visualizations now')
            visualize_data(data,ii,patchdim,[20,10])
            print('Saving SNR')
            plot_SNR(snr_list)
            print('Saving R_error')
            plot_residuals(residual_list)
            print('Average Coefficients')
            lbfgs_sc.plot_mean_firing(ii)
            print('Visualizations done....back to work now')
