import numpy as np
from numbapro import cuda
import numbapro.cudalib.cublas as cublas
from numba import *
from math import ceil

@cuda.jit('void(f4[:,:])')
def csub(c):
    n = c.shape[0]
    i = cuda.grid(1)
    
    if i<n:
        c[i,i] = 0.

@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,f4[:],f4,f4,i4)')
def iterate(c,b,ci,u,s,eta,thresh,lamb,adapt,softThresh):
    n = u.shape[0]
    m = u.shape[1]
    i,j = cuda.grid(2)
    
    if i<n and j< m:
        u[i,j] = eta*(b[i,j]-ci[i,j])+(1-eta)*u[i,j]
        if u[i,j] < thresh[i] and u[i,j] > -thresh[i]:
            s[i,j] = 0.
        elif softThresh == 1:
            if u[i,j] > 0.:
                s[i,j] = u[i,j]-thresh[i]
            else:
                s[i,j] = u[i,j]+thresh[i]
        else:
            s[i,j] = u[i,j]
        if thresh[i] > lamb:
            thresh[i] = thresh[i]*lamb

def infer(dictionary,stimuli,eta,lamb,nIter,adapt, coeffs=None, softThresh=0):
#Get Blas routines
    blas = cublas.Blas()
#Initialize arrays
    numDict = dictionary.shape[0]
    numStim = stimuli.shape[0]
    dataLength = stimuli.shape[1]
    u = np.zeros((numStim, numDict), dtype=np.float32, order='F')
    if coeffs is not None:
        u[:] = np.atleast_2d(coeffs)
    d_u = cuda.to_device(u)
    d_s = cuda.to_device(np.zeros((numStim,numDict),dtype=np.float32,order='F'))
    d_b = cuda.to_device(np.zeros((numStim,numDict),dtype=np.float32,order='F'))
    d_ci = cuda.to_device(np.zeros((numStim,numDict),dtype=np.float32,order='F'))
    d_c = cuda.to_device(np.zeros((numDict,numDict),dtype=np.float32,order='F'))
    
    #Move inputs to GPU
    d_dictionary = cuda.to_device(np.array(dictionary,dtype=np.float32,order='F'))
    d_stimuli = cuda.to_device(np.array(stimuli,dtype=np.float32,order='F'))

    blockdim2 = (32,32)
    blockdim1 = 32
    griddimcsub = int(ceil(numDict/blockdim1))
    griddimi = (int(ceil(numStim/blockdim2[0])),int(ceil(numDict/blockdim2[1])))
    
    #Calculate c: overlap of basis functions with each other minus identity
    blas.gemm('N','T',numDict,numDict,dataLength,1.,d_dictionary,d_dictionary,0.,d_c)
    csub[griddimcsub,blockdim1](d_c)
    blas.gemm('N','T',numStim,numDict,dataLength,1.,d_stimuli,d_dictionary,0.,d_b)
    thresh = np.mean(np.absolute(d_b.copy_to_host()),axis=1)
    d_thresh = cuda.to_device(thresh)
    #Update u[i] and s[i] for nIter time steps
    for kk in xrange(nIter):
        #Calculate ci: amount other neurons are stimulated times overlap with rest of basis
        blas.gemm('N','N',numStim,numDict,numDict,1.,d_s,d_c,0.,d_ci)
        iterate[griddimi,blockdim2](d_c,d_b,d_ci,d_u,d_s,eta,d_thresh,lamb,adapt,softThresh)
    u = d_u.copy_to_host()
    s = d_s.copy_to_host()
    return (s,u,thresh)
