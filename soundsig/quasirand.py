"""
Generate quasi-random numbers using Halton, Hammerseley, and Sobol sequences.
"""
from __future__ import division, print_function

import numpy as np

from soundsig.thirdparty.sobol_lib import i4_sobol_generate


def quasirand(M, N, type='sobol', spherical=False):
    """Generate N quasi-random M dimensional vectors.

        Args:
            M (int): the dimension of each random vector
            N (int): the number of random vectors to generate
            type (str): the type of sequence to generate ['sobol']
            spherical (bool): whether or not to generate points on a sphere (False)

        Returns:
            R (np.ndarray): an MxN matrix of quasi-random numbers.

    """

    Mused = M - int(spherical)

    R = None
    if type == 'sobol':
        R = i4_sobol_generate(Mused, N, 1)

    if spherical:
        # project M-1 dimensional random numbers onto an M dimensional sphere
        # taken from equation 3.2 of N. Rehman and D. P. Mandic "Multivariate empirical mode decomposition" (2014)

        #rescale the random numbers so they lie in (-pi, pi]
        pi_eps = 1e-6
        R *= 2*np.pi - pi_eps
        R -= np.pi - pi_eps

        # add extra dimension to end
        R = np.vstack([R, np.ones([N])])

        #update each dimension
        for k in reversed(range(M)):

            #construct the new row, which will be a product of sines and one cosine
            new_row = np.ones([N])
            for j in range(k):
                new_row *= np.sin(R[j, :])

            #multiply the last term, a cosine for all but the very last row
            if k < M-1:
                new_row *= np.cos(R[k, :])

            #assign the product of sines and cosines to the current row, which is not used in updating the rows
            #that come before it
            R[k, :] = new_row

    return R

