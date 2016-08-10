#
#
# Jesse Livezey 2014-04-19
#


import numpy as np

#Initialize settings for inference
def infer(basis, stimuli, eta, lamb, nIter, adapt, coeffs=None, softThresh=0):
    """Infers sparse coefficients for dictionary elements when representing a stimulus using LCA algorithm.

        Args:
            basis: Dictionary used to represent stimuli. Should be arranged along rows.
            coeffs: Values to start pre-threshold dictionary coefficients at for all stimuli.
            stimuli: Goals for dictionary representation. Should be arranged along rows.
            eta: Controls rate of inference.
            thresh: Threshold used in calculation of output variable of model neuron.
            lamb: Minimum value for thresh.
            nIter: Numer of times to run inference loop.
            softThresh: Boolean choice of threshold type.
            adapt: Amount to change thresh by per run.

        Results:
            s: Post-threshold dictionary coefficients.
            u: Pre-threshold internal *voltage.*
            thresh: Final value of thresh variable.
                                                                                                                            
        Raises:
        """
    numDict = basis.shape[0]
    numStim = stimuli.shape[0]
    dataSize = basis.shape[1]
    #Initialize u and s
    u = np.zeros((numStim, numDict))
    if coeffs is not None:
        u[:] = np.atleast_2d(coeffs)
    s = np.zeros_like(u)
    ci = np.zeros((numStim, numDict))

    # Calculate c: overlap of basis functions with each other minus identity
    c = basis.dot(basis.T) - np.eye(numDict)

    #b[i,j] is the overlap fromstimuli:i and basis:j
    b = stimuli.dot(basis.T)
    thresh = np.absolute(b).mean(1)
    #Update u[i] and s[i] for nIter time steps
    for kk in xrange(nIter):
        #Calculate ci: amount other neurons are stimulated times overlap with rest of basis
        ci[:] = s.dot(c)
        u[:] = eta*(b-ci)+(1-eta)*u
        if softThresh == 1:
            s[:] = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis]) 
        else:
            s[:] = u
            s[np.absolute(s) < thresh[:,np.newaxis]] = 0.
        thresh[thresh>lamb] = adapt*thresh[thresh>lamb]
    return (s,u,thresh)
