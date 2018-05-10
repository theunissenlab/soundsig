from __future__ import division, print_function

import numpy as np


def compute_R2(x, y):
    return(((x - x.mean()) * (y - y.mean())).mean() / (x.std()*y.std()))**2


def effective_dof(X, lambda_val):
    """ Compute the effective degrees-of-freedom for data matrix X and lambda value. """
    s = np.linalg.svd(X, compute_uv=False)
    return np.sum(s**2 / (s**2 + lambda_val))
