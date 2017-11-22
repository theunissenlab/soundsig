"""
    Implementation of multi-variate EMD according to:
    N Rehman and DP Mandic "Multivariate empirical mode decomposition"
    Proc. R. Soc. A (2010) 466, 1291-1302 doi:10.1098/rspa.2009.0502



"""
from __future__ import division, print_function

import copy
import numpy as np
from scipy.interpolate import splrep,splev
from scipy.stats import pearsonr
import time
from soundsig.quasirand import quasirand
from soundsig.signal import find_extrema


def create_mirrored_spline(mini, maxi, s):
    """
        To reduce end effects, we need to extend the signal on both sides and reflect the first and last extrema
        so that interpolation works better at the edges
    """

    #get the values of s at the minima and maxima
    s_min = list(s[mini])
    s_max = list(s[maxi])

    #reflect the extrema on the left side
    Tl = maxi[0]  # index of left-hand (first) maximum
    tl = mini[0]  # index of left-hand (first) minimum

    maxi.insert(0, -tl)
    s_max.insert(0, s_max[0])
    mini.insert(0, -Tl)
    s_min.insert(0, s_min[0])

    #reflect the extrema on the right side
    T = len(s)
    Tr = maxi[-1]  # index of right hand (last) maximum
    tr = mini[-1]  # index of right hand (last) minimum

    maxi.append((T-tr) + T)
    s_max.append(s_max[-1])
    mini.append((T-Tr) + T)
    s_min.append(s_min[-1])

    if len(maxi) < 4 or len(mini) < 4:
        return None, None

    #interpolate the upper and lower envelopes
    upper_env_spline = splrep(maxi, s_max, k=3)
    lower_env_spline = splrep(mini, s_min, k=3)

    return lower_env_spline,upper_env_spline


def compute_mean_envelope(s, nsamps=1000):
    """ Use random sampling to compute the mean envelope of a multi-dimensional signal.

    Args:
        s (np.ndarray): an NxT matrix describing a multi-variate signal. N is the number of channels, T is the number of time points.
        nsamps (int): the number of N dimensional projections to use in computing the multi-variate envelope.

    Returns:
        env (np.ndarray): an NxT matrix giving the multi-dimensional envelope of s.
    """

    N,T = s.shape

    #pre-allocate the mean envelope matrix
    mean_env = np.zeros([N, T])

    #generate quasi-random points on an N-dimensional sphere
    #stime = time.time()
    R = quasirand(N, nsamps, spherical=True)
    #etime = time.time() - stime
    #print 'Elapsed time for quasirand: %d seconds' % int(etime)

    stime = time.time()
    for k in range(nsamps):
        #istime = time.time()
        r = R[:, k].squeeze()

        #print 'k=%d, s.shape=%s, r.shape=%s' % (k, str(s.shape), str(r.shape))

        #project s onto a scalar time series using random vector
        #dtime = time.time()
        p = np.dot(s.T, r)
        #detime = time.time() - dtime
        #print '\t[%d] Time to do dot %0.6f s' % (k, detime)

        #print 'p.shape=',p.shape

        #identify minima and maxima of projection
        #eetime = time.time()
        mini_p,maxi_p = find_extrema(p)
        #eeetime = time.time() - eetime
        #print '\t[%d] time to find extrema: %0.6fs' % (k, eeetime)

        #for each signal dimension, fit maxima with cubic spline to produce envelope

        t = np.arange(T)
        for n in range(N):
            #sptime = time.time()
            mini = copy.copy(mini_p)
            maxi = copy.copy(maxi_p)
            if len(mini) < 4 or len(maxi) < 4:
                return None

            #extrapolate edges using mirroring
            lower_env_spline, upper_env_spline = create_mirrored_spline(mini, maxi, s[n, :].squeeze())
            if lower_env_spline is None or upper_env_spline is None:
                return None

            #evaluate upper and lower envelopes
            upper_env = splev(t, upper_env_spline)
            lower_env = splev(t, lower_env_spline)

            #compute the envelope for this projected dimension
            env = (upper_env + lower_env) / 2.0

            #update the mean envelope for this dimension in an online way
            delta = env - mean_env[n, :]
            mean_env[n, :] += delta / (k+1.0)
            #esptime = time.time() - sptime
            #print '\t[%d] time for spline iteration on dimension %d: %0.6fs' % (k, n, esptime)
        #ietime = time.time() - istime
        #print '\t[%d] took %0.6f seconds' % (k, ietime)

    etime = time.time() - stime
    #print '%d samples took %0.6f seconds' % (nsamps, etime)

    return mean_env


def sift(s, nsamps=100, resolution=50.0, max_iterations=30, verbose=False):
    """Do a single iteration of multi-variate empirical mode decomposition (MEMD) on the multi-dimensional signal s, obtaining a multi-variate IMF.

    Args:
        s (np.ndarray): an NxT matrix describing a multi-variate signal. N is the number of channels, T is the number of time points.
        nsamps (int): the number of N dimensional projections to use in computing the multi-variate envelope.
        resolution (float): the maximum log10 ratio of the initial signal energy to average envelope energy, used as stopping criteria (Rato et. al 2008, sec 3.2.3)
        max_iterations (int): the maximum number of iterations before quitting

    Returns:
        imf (np.ndarray): an NxT matrix giving the multi-dimensional IMF for this sift.

    """

    converged = False

    # inintialize the residual signal to the signal
    N,T = s.shape
    r = copy.copy(s)

    initial_energy = (s**2).mean()
    avg_envelope_energy = 0.0
    iteration = 1
    while not converged:
        # compute the mean envelope
        env = compute_mean_envelope(r, nsamps=nsamps)
        if env is None:
            converged = True
            break

        #update the average envelope energy
        env_energy = (env**2).mean()
        avg_envelope_energy += (env_energy - avg_envelope_energy) / iteration

        # compute the fraction of the mean_envelope that will be subtracted from the residual, meant to minimize the
        # energy of the signal (Rato et. al 2008, sec 3.2.3)
        alpha = np.zeros([N])
        for k in range(N):
            a,p = pearsonr(r[k, :], env[k, :])
            if np.isnan(a):
                print('[iter %d] alpha=NaN for dimension %d!' % (iteration, k))
                a = 1e-2
            alpha[k] = max(a, 1e-2)

        # subtract the mean envelope from the residual
        final_alpha = alpha.mean()
        r -= alpha.mean()*env

        # test the residual for convergence to an IMF using stoppage criteria

        #compute the "resolution factor", the ratio of initial energy to average envelope energy. the higher the
        #"resolution", the lower the average envelope energy, meaning that the IMF is converging.
        resolution_factor = np.log10(initial_energy / avg_envelope_energy)

        if verbose:
            print('sift iter %d: initial_energy=%0.3f, env_energy=%0.3f, avg_envelope_energy=%0.3f, final_alpha=%0.2f, resolution_factor=%0.2f' %
                  (iteration, initial_energy, env_energy, avg_envelope_energy, final_alpha, resolution_factor))

        if resolution_factor > resolution:
            converged = True
        if iteration >= max_iterations:
            converged = True

        iteration += 1

    # after repeatedly subtracting off the mean envelope, we are left with the intrinsic mode function (IMF), which
    # basically contains the higher frequency components of the original signal, with the lower frequency components
    # subtracted off in a way that deals well with nonstationarity
    return r


def memd(s, nimfs, nsamps=100, resolution=1.0, max_iterations=30, num_noise_channels=0, verbose=False):

    imfs = list()
    N,T = s.shape
    if verbose:
        print('Starting MEMD: N=%d, T=%d, nimfs=%d, nsamps=%d, resolution=%0.2f, max_iterations=%d, num_noise_channels=%d' %
                        (N, T, nimfs, nsamps, resolution, max_iterations, num_noise_channels))
    r = copy.copy(s)

    if num_noise_channels > 0:
        #add gaussian noise channels for Noise-assisted MEMD (2011 Rehnert and Mandic)
        noise = np.random.randn(num_noise_channels, T)
        r = np.vstack([r, noise])

    for n in range(nimfs):
        imf = sift(r, nsamps=nsamps, resolution=resolution, max_iterations=max_iterations, verbose=verbose)
        imfs.append(imf)
        r -= imf
    imfs = np.array(imfs)

    return imfs[:, :N, :]
