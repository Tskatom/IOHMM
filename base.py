"""
Created on Feb 13 2015
@author Wei Wang

Implement the base class for IOHMM
The implementation is based on:
    - Input-Output HMM's for Sequence processing
    - scikit-learn HMM implementation
"""
__author__ = 'weiwang'
__email__ = 'tskatom@vt.edu'

from __future__ import division
import string
import numpy as np
from .util import extmath

ZEROLOGPROB = -1e200
EPS = np.finfo(float).eps
NEGINF = -np.inf

class _BaseIOHMM():
    """
    Input output Hidden Markov model base class
    Representation of a IO-HMM model probability distribution.
    This class allows for sampling from, evaluation and maximum-likelihood
    estimation of the parameters of a HMM

    Attributes
    ----------
    n_components : int
        Numbers of the states in the model

    ins : list
        list of array-like observation sequences, each of which has shape
                (n_i, n_features), which n_i is the length of the i_th observation

    startprob : array, shape ('n_components')
        Initiate state occupation distribution
        
    algorithm : string, one of the decoder_algorithms
        Decoder Algorithms

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance

    n_iter : int, optional
        Number of iterations to perform

    thresh : float, optional
        Convergence threshold

    params: string, optional
        Controls which parameters are updated in the training process.

    init_params: string, optional
        Controls which parameters are initiated prior to training

    Interfaces
    ----------
    This class implements the public interface for all IOHMMs that derived from it,
    including all of the machinery for the forward-backward Viterbi algorithm. Subclass
    only need to implement _generate_sample_from_state(), _compute_likelihood(),
    _init(), _initiatelize_sufficient_statistics() and _do_mstep()
    """

    def __init__(self, n_components, ins, startprob=None, algorithm="viterbi",
                 random_state=None, n_iter=0, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):

        self.n_components = n_components
        self.ins = ins
        self.n_iter = n_iter
        self.thresh = thresh
        self.params = params
        self.init_params = init_params
        self.startprob = startprob
        self.algorithm = algorithm
        self.random_state = random_state
        self.input_dim = ins[0].shape[1]  # the dimension of input

        # construct the transition weighted matrix
        self.trans_weight_mat = np.tile(1/(self.input_dim + 1),
                                        (self.n_components, self.n_components, self.input_dim + 1))


    def fit(self, obs):
        """ Estimate the model parameters

        An initialization step is performed before enter te EM algorithm
        :param obs:
            obs : list
                list of array-like observation sequences, each of which has shape
                (n_i, n_features), which n_i is the length of the i_th observation

        """

        self._init(obs, self.init_params)  # initiate the model
        logprob = []
        for i in range(self.n_iter):
            # Expectation step
            stats = self._initialize_sufficient_statistics()

            for i in range(len(obs)):
                obs_seq = obs[i]
                ins_seq = self.ins[i]

                transmat = self._compute_transmat(ins_seq) # compute dynamic transition matrix with shape (t, n, n)
                framelogprob = self._compute_likelihood(ins_seq, obs_seq)  # compute p(y|U, x_t=i) with shape (t, n)

                lpr, fwdlattice = self._do_forward_pass(transmat, framelogprob, ins_seq)
                bwdlattice = self._do_backward_pass(transmat, framelogprob, ins_seq)


    def _compute_transmat(self, ins_seq):
        """ Compute the dynamic transition weight matrix for each time step"""
        #initiate the dynamic transition matrix
        transmat = np.tile(0.0, (len(ins_seq, self.n_components, self.n_components)))
        for t in range(len(ins_seq)):
            u = ins_seq[t][np.newaxis].T  # transform u into column vector
            for i in range(self.n_components):
                weightMat = self.trans_weight_mat[i]
                prob = extmath.softmax(np.dot(weightMat, u))
                transmat[i] = prob
        return transmat

    def _do_forward_pass(self, transmat, framelogprob, ins_seq):
        """  Compute the forward lattice
        :param transmat:
        :param framelogprob:
        :param ins_seq:
        :return: p(obs_seq|ins_seq) and p(x_t=i, y_(1:t)|u_(1:t))
        """
        T = len(ins_seq)
        fwdlattice = np.zeros((T, self.n_components))
        scaling_factors = np.zeros(T)
        fwdlattice[0] = np.dot(transmat[0].T, self.startprob[np.newaxis].T).flatten() * framelogprob[0]
        scaling_factors[0] = 1 / np.sum(fwdlattice[0])
        fwdlattice[0] = fwdlattice[0] * scaling_factors[0]

        for t in range(1, T):
            fwdlattice[t] = np.dot(transmat[0].T, fwdlattice[t-1][np.newaxis].T).flatten() * framelogprob[t]
            scaling_factors[t] = 1 / np.sum(fwdlattice[t])
            fwdlattice[t] = fwdlattice[t] * scaling_factors[t]


        return np.exp(-1*np.sum(np.log(scaling_factors[-1]))), fwdlattice



    def _init(self, obs, params):
        """
        Initialize the model parameters

        :param ins:
        :param obs:
        :param params:
        :return:
        """""
        pass

    def _initialize_sufficient_statistics(self):
        # Initialize the sufficient statistics
        pass






