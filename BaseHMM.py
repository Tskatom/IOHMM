"""
Created on Dec 15 2014
@author Wei Wang

Implement the base class for iohmm
The implementation is based on:
    - Input-Output HMM's for Sequence processing
    - Guyz's hmm implementation in https://bitbucket.org/GuyZ/hmm
"""
__author__ = 'weiwang'
__email__ = 'tskatom@vt.edu'


import numpy as np
from util import logs

logs.init()
logger = logs.getLogger(__name__)

class BaseHMM(object):
    """
    In this circumstance, we use u represent input node, x represent hidden node and
     y represent the output node
    """
    def __init__(self, n, precision=np.double, verbose=False):
        self.n = n
        self.verbose = verbose
        self.precision = precision

    def _calcalpha(self, observations):
        """
        compute the forward variable alpha(T * N), T is the length of the chain,
        N is the number of hidden status,
        alpha[t][i] = p(y(1:t), x(t)=i|u(1:t))
        = p(y(t)|x(t)=i, u(t)) * sum[(phi_il(ut)) * alpha[t-1][i]]
        B_map[i][t] = p(y(t)|x(t)=i, u(t))
        A_map[t][i][j] represent p(x(t)=j|u(t), x(t-1)=i)
        """
        alpha = np.zeros((len(observations), self.n), dtype=self.precision)
        # initiate the alpha
        for i in range(self.n):
            alpha[0][i] = self.B_map[i][0] * self.A0_map[0][i]

        # the remaining time interation
        for t in range(1, len(observations)):
            for i in xrange(self.n):
                for j in xrange(self.n):
                    alpha[t][i] += self.A_map[t][j][i] * alpha[t-1][j]
                alpha[t][i] = alpha[t][i] * self.B_map[i][t]
        return alpha

    def _calcbeta(self, observations):
        """
        compute the backward variable beta(T * N).
        beta[t][i] = p(y(t+1;T)|x(t)=i, u(1:T))
        beta[t][i] = sum(p(y(t+1)|x(t+1)=l)*phi(li,t)*beta(t)[l])
        """
        T = len(observations)
        beta = np.zeros((T, self.n), dtype=self.precision)
        # initiate beta[T][i]
        beta[T-1, :] = np.ones(self.n)

        # update the remaining beta
        for t in xrange(T-2, -1, -1):
            beta[t,:] = np.zeros(self.n)
            for i in xrange(self.n):
                for j in xrange(self.n):
                    beta[t][i] += self.B_map[j][t+1] * self.A_map[t+1][i][j] * self.B_map[t+1][j]
        return beta

    def likelihood(self, observations, cache=False):
        """
        compute the likelihood of the observation p(y(1:T)|u(1:T))
        using forwardbackward algorithm
        """
        if not cache:
            self._mapB(observations)
        alpha = self._calcalpha(observations)
        likelihood = np.log(np.sum(alpha[-1]))
        return likelihood

    def decode(self, observations):
        """
        find the most probable hidden sequence given whole input and output sequence
        hidden_path = argmax(p(x(1:T)|u(1:T), y(1:T)))
        """
        pass

    def predict(self, observations):
        """
        compute one step probability  p(y(t+1)|u(1:T), y(1:T)),
        then output E(y(t+1)) as one step prediction
        """
        raise NotImplementedError("A prediction inference function must be implemented")

    def _calcHij(self, observations):
        """
        compute the transition posterior joint probability H[t][i][j] = p(x(t)=i, x(t-1)=j|y(1:T), u(1:T))
        which is used in training process
        """
        T = len(observations)
        h = np.zeros((T, self.n, self.n), dtype=self.precision)
        self._mapB(observations)
        alpha = self._calcalpha(observations)
        beta = self._calcbeta(observations)
        likelihood = self.likelihood(observations)
        for t in xrange(T-1, 0):
            for i in xrange(self.n):
                for j in xrange(self.n):
                    h[t][i][j] = self.B_map[t][i] * alpha[t-1][i] * beta[t][i] * self.A_map[t][j][i] / likelihood

        return h

    def _calcGi(self, observations):
        """
        compute the state posterior probability p(x(t)=i|y(1:T), u(1:T))
        which is used in training process
        """
        T = len(observations)
        g = np.zeros((T, self.n), dtype=self.precision)
        alpha = self._calcalpha(observations)
        beta = self._calcbeta(observations)
        likelihood = self.likelihood(observations)
        for t in range(T):
            for i in xrange(self.n):
                g[t][i] = alpha[t][i] * beta[t][i] / likelihood
        return g

    def _calcstatis(self, observations):
        """
        compute the sufficient statics for e step in EM algorithm for IOHMM
        phi(ij, t): p(x(t)=j|x(t-1)=i, u(t))
        n(j, t): E(y(t)|x(t)=j, u(t))

        alpah(i, t): p(y(1:t), x(t)=i|u(1:t))
        beta(j, t): p(y(t+1:T)|x(t)=j, u(1:T))

        h(ij, t): p(x(t)=i, x(t-1)=j| y(1:T), u(1:T))
        g(i, t): p(x(t)=i|y(1:T), u(1:T))
        """
        stats = {}
        stats['alpha'] = self._calcalpha(observations)
        stats['beta'] = self._calcbeta(observations)
        stats['g'] = self._calcGi(observations)
        stats['h'] = self._calcHij(observations)
        return stats


    def train(self, observations, iterations=10, epsilon=0.0001, threshold=-0.001):
        """
        Update the HMMs parameters given a set of observation sequences

        The training process will run 'iterations' times, or until log likelihood of the model
        increase by less than epsilon

        Threshold denotes the algorithms sensitivity to the log likelihood decreasing from one
        iteration to the next
        """
        self._mapB(observations)

        for i in xrange(iterations):
            likelihood_old, likelihood_new = self.trainiter(observations)
            if self.verbose:
                logger.info("Iter %d Old Likelihood = %0.4f New Likelihood = %0.4f converging=" %
                            (i, likelihood_old, likelihood_new, (likelihood_new - likelihood_old) > threshold))

            if abs(likelihood_new - likelihood_old) <= epsilon:
                break  # converged

    def trainiter(self, observations):
        """
        A single iteration of an EM algorithm
        return the old likelihood and the one for the new model
        """
        # call the EM algorithm
        new_model = self._baumwelch(observations)

        # calculate the likelihood of the old model
        likelihood_old = self.likelihood(observations)

        # update the model with the new estimation
        self._updatemodel(new_model)

        # calculate the new likelihood
        likelihood_new = self.likelihood(observations);

        return likelihood_old, likelihood_new

    def _updatemodel(self, new_model):
        """
        replace the old model with the new one
        """
        self.model = new_model

    def _reestimate(self, stats, observations):
        """
        The deriving classing should override/extend this method to calculate
        additonal parameters
        """
        raise NotImplementedError("Estimate the parameters in the model")


    def _baumwelch(self, observations):
        """
        The EM algorithm to learn the parameter of the HMM model
        """
        # E step calculate the statistics
        stats = self._calcstatis(observations)

        # M step re-estimate the parameters
        return self._reestimate(stats, observations)


    def _mapB(self, observations):
        """
        Compute the PDF for each observation in the sequence based on hidden status and input, each
        deriving class should implement this method

        This method does not explicitly return a value, but it expects that self.B_map is internally
        computed as mentioned above. self.B_map is an T * N numpy array
        """
        raise NotImplementedError("a mapping function for B(observable probabilities) must be implemented")

    def _mapT(self, observations):
        """
        Compute the transition probability
        """
        raise NotImplementedError("a mapping function for A(transition probabilities) must be implemented")