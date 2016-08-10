from __future__ import print_function
import numpy as np

from LCAversions.LCAnumpy import lca as lcan
from LCAversions.LCAnumbaprog import lca as lcag
from LCAversions.LCAfortran import lca as lcaf
from LCAversions.LCAmomentum import lca as lcam

class test_infer():

    def setup(self):
        self.rng = np.random.RandomState(0)
        self.num = 64
        self.numDict = 1024
        self.numStim = 64
        self.dataSize = 128
        self.nIter = 500
        self.eta = .05
        self.lamb = .05
        self.adapt = .1
        self.softThresh = 0

    def test_numbaprog(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,dtype=np.float32,order='F')
        stimuli = np.array(stimuli,dtype=np.float32,order='F')
        s,u,thresh = lcag.infer(dictionary,
                                stimuli,
                                self.eta,
                                self.lamb,
                                self.nIter,
                                self.adapt)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,dtype=np.float32,order='F')
        stimuli = np.array(stimuli,dtype=np.float32,order='F')
        s,u,thresh = lcag.infer(dictionary,
                                stimuli,
                                self.eta,
                                self.lamb,
                                self.nIter,
                                self.adapt)
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)

    def test_fortran(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,order='F')
        coeffs = np.array(coeffs,order='F')
        stimuli = np.array(stimuli,order='F')
        s = np.zeros_like(coeffs,order='F')
        u = np.zeros_like(coeffs,order='F')
        thresh = np.zeros(self.num,order='F')
        lcaf.lca(dictionary,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt,s,u,thresh,self.num,self.num,self.num)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        coeffs = np.zeros(shape=(self.numStim,self.numDict))
        #Change dtype and enforce Fortran ordering
        dictionary = np.array(dictionary,order='F')
        coeffs = np.array(coeffs,order='F')
        stimuli = np.array(stimuli,order='F')
        s = np.zeros_like(coeffs,order='F')
        u = np.zeros_like(coeffs,order='F')
        thresh = np.zeros(self.numStim,order='F')
        try:
            lcaf.lca(dictionary,stimuli,self.eta,self.lamb,self.nIter,self.softThresh,self.adapt,s,u,thresh,self.numDict,self.numStim,self.dataSize)
        except:
            print('Looks like LCA Fortran implementation is not properly compiled. Peacing out of this one')
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)

    def test_numpy(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        s,u,thresh = lcan.infer(dictionary,
                                stimuli,
                                self.eta,
                                self.lamb,
                                self.nIter,
                                self.adapt)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        coeffs = np.zeros(shape=(self.numStim,self.numDict))
        s,u,thresh = lcan.infer(dictionary,
                                stimuli,
                                self.eta,
                                self.lamb,
                                self.nIter,
                                self.adapt)
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)

    def test_momentum(self):
        coeffs = np.zeros(shape=(self.num,self.num))
        #Test for correct outputs for simple data
        dictionary = np.diag(np.ones(self.num))
        stimuli = np.diag(np.ones(self.num))
        s,u,thresh = lcam.infer(dictionary,
                                stimuli,
                                self.eta,
                                self.lamb,
                                self.nIter,
                                self.adapt)
        assert np.allclose(s,np.diag(np.ones(self.num)))
        assert np.allclose(u,np.diag(np.ones(self.num)))
        #Test on random data
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T)))).dot(dictionary)
        stimuli = self.rng.randn(self.numStim,self.dataSize)
        s,u,thresh = lcam.infer(dictionary,
                                stimuli,
                                self.eta,
                                self.lamb,
                                self.nIter,
                                self.adapt)
        assert np.allclose(stimuli,s.dot(dictionary),atol=1e-5)
