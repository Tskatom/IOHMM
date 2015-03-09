import numpy as np
from scipy.optimize import minimize

def res(x, j):
    obj = (j-2) * x * x
    return obj

from functools import partial
pf = partial(res, j=1)
print pf(x=2)
