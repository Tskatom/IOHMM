from __future__ import division
__author__ = 'weiwang'
__mail__ = 'tskatom@vt.edu'


from base import _BaseIOHMM
import numpy as np

if __name__ == '__main__':
    T = 2
    n_components = 2
    transmat = np.tile(0.5, (3, 2, 2))
    probys = np.tile(0.2, (3, 2))
    startprob = np.array([0.4, 0.6])
    ins = np.tile([1.0, 1.0], (3, 1))
    hmm = _BaseIOHMM(n_components, [ins], startprob=startprob)
    likelihood, fwdlattice = hmm._do_forward_pass(transmat, probys)
    print likelihood
    print fwdlattice

    print 'Scalfactor', hmm.scaling_factors
    bwdlattice = hmm._do_backward_pass(transmat, probys)
    print bwdlattice