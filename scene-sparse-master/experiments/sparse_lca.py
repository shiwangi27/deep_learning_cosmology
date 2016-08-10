#This script will run Sparse coding on Aude Oliva's new database that is
#an extension of the SUN database
import importlib
import tables
import numpy as np
import random
from LCAversions.LCAnumpy import lca as lcanump 
from LCAversions.LCAnumbaprog import lca as lcanumb
import pdb

def load_list(filename_no_extension):
    """
    :type filename_no_extension: A filename string without the .py extension
    """
    print('loading file ' + filename_no_extension)
    module = importlib.import_module(filename_no_extension)
    assert isinstance(module, object)
    assert isinstance(module.content, list)
    return module.content

#Update basis
def update_basis(basis, coeff, LR,data): 
    Residual = data - np.dot(basis,coeff.T)
    dbasis = LR * (np.dot(Residual,coeff))
    basis = basis + dbasis
    #Normalize basis
    norm_basis = np.diag(1/np.sqrt(np.sum(basis**2,axis=0)))
    basis = np.dot(basis,norm_basis)
    print('The norm of the basis is',np.linalg.norm(basis))
    return basis,residual

def save_basis(basis,path,iterations):

    return 1

#This be main function
def sparse_lca(basis=None,coeff=None,batch=50,LR=0.05,lamb=0.05,eta=.01,iter=10,InferIter=300,adapt=0.99,patchdim=32):
    indoor_file_list = load_list('indoor_file_list')
    outdoor_file_list = load_list('outdoor_file_list')
    if basis is None and coeff is None:
        basis = np.random.randn(patchdim**2,(patchdim**2)*4)
        coeff = np.random.randn(2*batch,(patchdim**2)*4)
    elif coeff is None:
        coeff = np.random.randn(2*batch,(patchdim**2)*4)
    elif basis is None:
        basis = np.random.randn(patchdim**2,(patchdim**2)*4)

    # to get the image-data from the hdf5 file:
    #Let's create a file handle
    try:
        h = tables.open_file('/media/mudigonda/Gondor/Data/scene-sparse/places32.h5','r')
    except:
        print('Could not open file handle')
        return 1

    #Modify this to get a batch of images
    #Generate two random numbers corresponding to the len(list)-batchsize
    for ii in range(iter):
        indoor_idx=random.randint(0,len(indoor_file_list)-batch-1)
        outdoor_idx=random.randint(0,len(outdoor_file_list)-batch-1)
        print(indoor_idx,outdoor_idx)
        print('The range of the batch is then')
        print(indoor_idx+batch,outdoor_idx+batch)
        data = np.zeros([patchdim**2,2*batch])
        counter=-1
        for idx in range(batch): 
            image_indoor = h.root._f_get_child(indoor_file_list[indoor_idx+idx])[:]
            image_outdoor = h.root._f_get_child(outdoor_file_list[outdoor_idx+idx])[:]
            counter +=1
            data[:,counter] = image_indoor.flatten()
            counter +=1
            data[:,counter] = image_outdoor.flatten()
        #Infer Coefficients (on updated dictionary)
        [coeff,u,thresh] = lcanumb.infer(basis.T,data.T,eta,lamb,InferIter,adapt)
        if np.any(np.isnan(coeff)):
            print('coeff has nans')
            pdb.set_trace()
        if np.linalg.norm(coeff)==0:
            print('No active coeff, also a doo doo')
            pdb.set_trace()
        #Update basis (dictionary)
        [basis,residual]=update_basis(basis,coeff,LR,data)
        if np.any(np.isnan(basis)):
            print('basis has nans')
            pdb.set_trace()
        if np.linalg.norm(basis)==0:
            print('all basis be zero. that is a doo doo')
            pdb.set_trace()
        #For every x 1000 iterations, save the basis
        print('Residual Error, Norm is ', np.linalg.norm(residual))

    #Close the file handle
    h.close()
    return 1
