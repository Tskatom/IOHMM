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
