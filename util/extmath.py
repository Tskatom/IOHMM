__author__ = 'Wei Wang'
__email__ = "tskatom@vt.edu"

from __future__ import division
import numpy as np


def softmax(a):
    return np.exp(a) / sum(np.exp(a))
