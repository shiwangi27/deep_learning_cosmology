#This file will time various versions of LCA
from __future__ import division
import numpy as np
import sklearn.preprocessing as skp
from timeit import default_timer as timer

from LCAnumpy import lca as lcan
from LCAfortran import lca as lcaf
from LCAnumbaprog import lca as lcag

def main():
    """Profiles various versions of LCA."""

    nshort = 6
    tshort = 2
    nmed = 3
    tmed = 6
    nlong = 1
    
    #Setup variables for inference
    numDict = int(2048)
    numBatch = int(128)
    dataSize = int(256)
    dictsIn = np.random.randn(numDict,dataSize)
    # LCA requires that dictionary be unit norm
    dictsIn = skp.normalize(dictsIn, axis=1)
    stimuli = np.random.randn(numBatch,dataSize)
    batchCoeffs = np.random.randn(numBatch,numDict)
    coeffs = np.zeros((numBatch, numDict))
    eta = .01
    lamb = .05
    nIter = 300
    adapt = .99
    softThresh = 0
    thresh = np.random.randn(numBatch)
    
    #LCA
    params = """Parameters:
             numDict: """+str(numDict)+"""
             numBatch: """+str(numBatch)+"""
             dataSize: """+str(dataSize)+"""
             nIter: """+str(nIter)+"""\n"""
    print params
             
    start = timer()
    lcan.infer(dictsIn,stimuli,eta,lamb,nIter,adapt)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcan.infer(dictsIn,stimuli,eta,lamb,nIter,adapt)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '---------------Numpy based LCA----------------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt

    dictsIn = np.array(dictsIn,order='F')
    stimuli = np.array(stimuli,order='F')
    coeffs = np.array(coeffs,order='F')
    batchCoeffs = np.array(batchCoeffs,order='F')
    thresh = np.array(thresh,order='F')

    start = timer()
    lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcaf.lca(dictsIn,stimuli,eta,lamb,nIter,softThresh,adapt,coeffs,batchCoeffs,thresh,numDict,numBatch,dataSize)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '---------------Fortran based LCA--------------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt

    dictsIn = np.array(dictsIn,dtype=np.float32,order='F')
    stimuli = np.array(stimuli,dtype=np.float32,order='F')
    start = timer()
    lcag.infer(dictsIn,stimuli,eta,lamb,nIter,adapt)
    dt = timer()-start
    if dt < tshort:
        n_times = nshort
    elif dt < tmed:
        n_times = nmed
    else:
        n_times = nlong
    for ii in xrange(n_times-1):
        start = timer()
        lcag.infer(dictsIn,stimuli,eta,lamb,nIter,adapt)
        dt = dt+timer()-start
    dt = dt/(n_times)
    print '----------------GPU based LCA-----------------'
    print 'Average time over '+str(n_times)+' trials:'
    print '%f s' % dt

if __name__ == '__main__':
    main()
