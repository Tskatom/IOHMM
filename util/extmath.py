from __future__ import division
__author__ = 'Wei Wang'
__email__ = "tskatom@vt.edu"


import numpy as np
from scipy.misc import factorial
import sys

def softmax(a):
    return np.exp(a) / sum(np.exp(a))

def poisson_likelihood(lam, y):
    prob = np.multiply(np.exp(-1 * lam), np.power(lam, y)) / factorial(y)
    return prob

def log_sum(theta, u):
    # u is a D by 1 column vector
    return np.log(np.sum(np.exp(np.dot(theta, u))))

def logsumexp(x):
    _max = max(x)
    pow_sum = 0.0
    for i in range(len(x)):
        pow_sum += np.exp(x[i] - _max)
    return np.log(pow_sum) + _max

def logfac(y):
    if y == 0:
        return 0
    return np.sum(np.log(np.arange(1.0, y + 1)))

def log_poisson_likelihood(lam, y):
    tmp_y = np.squeeze(np.array(y))
    log_fact =np.zeros(len(tmp_y))
    for i in range(len(log_fact)):
        log_fact[i] = logfac(tmp_y[i])
    log_prob = -1 * lam + np.multiply(y, np.log(lam)) - np.matrix(log_fact).T
    return log_prob

def safe_softmax(x):
    v_max = np.max(x, axis=1)
    tmp_x = x - v_max
    return np.exp(tmp_x)/np.sum(np.exp(tmp_x), axis=1)

if __name__ == "__main__":
     x = np.array([1,2,3])
     print logsumexp(x)