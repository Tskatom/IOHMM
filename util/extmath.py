from __future__ import division
__author__ = 'Wei Wang'
__email__ = "tskatom@vt.edu"


import numpy as np
import math

def softmax(a):
    return np.exp(a) / sum(np.exp(a))

def poisson_likelihood(lam, y):
    prob = (np.exp(-1 * lam) * np.power(lam, y)) / math.factorial(y)
    return prob

def log_sum(theta, u):
    # u is a D by 1 column vector
    return np.log(np.sum(np.exp(np.dot(theta, u))))
