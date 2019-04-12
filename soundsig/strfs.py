from __future__ import division, print_function

import time
import numpy as np
from sklearn.linear_model import Ridge


def strf_correlation(strf1, strf2, max_delay=10):
    """
        Computes the dot product between two STRFs across in a way that's invariant to frequency differences and time lags up to max_delay.
    """

    if strf1.shape != strf2.shape:
        raise ValueError('STRF shapes do not match: %s != %s' % (str(strf1.shape), str(strf2.shape)))
    nf = strf1.shape[0]

    time_shifts = np.arange(-max_delay, max_delay+1, 1, dtype='int')
    freq_shifts = np.arange(nf, dtype='int')
    all_ccs = np.zeros([len(time_shifts), len(freq_shifts)])
    s1 = strf1 / np.abs(strf1).max()
    s2 = strf2 / np.abs(strf2).max()
    ns1 = s1 - s1.mean()
    for i,d in enumerate(time_shifts):
        for j,f in enumerate(freq_shifts):
            #roll along frequency axis
            rs2 = np.roll(s2, f, axis=0)
            #shift and zero along time axis
            rs2 = np.roll(rs2, d, axis=1)
            if d > 0:
                rs2[:, :d] = 0.0
            elif d < 0:
                rs2[:, d:] = 0.0
            ns2 = rs2 - rs2.mean()
            cc = (ns1 * ns2).mean() / ( s1.std() * s2.std() )
            all_ccs[i, j] = cc

    all_ccs_abs = np.abs(all_ccs)
    mi = np.unravel_index(all_ccs_abs.argmax(), all_ccs.shape)
    argmax_delay,argmax_freq = time_shifts[mi[0]], freq_shifts[mi[1]]
    return argmax_delay, argmax_freq, all_ccs[mi[0], mi[1]]


def strf_mps(strf, fstep, sample_rate, half=False):

    nchannels,strflen = strf.shape
    fstrf = np.fliplr(strf)
    mps = np.fft.fftshift(np.fft.fft2(fstrf))
    amps = np.real(mps * np.conj(mps))

    #Obtain labels for frequency axis
    dwf = np.zeros([nchannels])
    fcircle = 1.0 / fstep
    for i in range(nchannels):
        dwf[i] = (i/float(nchannels))*fcircle
        if dwf[i] > fcircle/2.0:
            dwf[i] -= fcircle

    dwf = np.fft.fftshift(dwf)
    if dwf[0] > 0.0:
        dwf[0] = -dwf[0]

    #Obtain labels for time axis
    fcircle = sample_rate
    dwt = np.zeros([strflen])
    for i in range(strflen):
        dwt[i] = (i/float(strflen))*fcircle
        if dwt[i] > fcircle/2.0:
            dwt[i] -= fcircle

    dwt = np.fft.fftshift(dwt)
    if dwt[0] > 0.0:
        dwt[0] = -dwt[0]

    if half:
        halfi = np.where(dwf == 0.0)[0][0]
        amps = amps[halfi:, :]
        dwf = dwf[halfi:]

    return dwf,dwt,amps


def strf_conv(X, strf, lags, bias):
    """ Performs a convolution between the multivariate time series "input" with the spatio-temporal receptive
        field ("strf").

        :param X: The input time series, of shape (num_time_points,num_features)
        :param strf: The receptive field, of shape (num_features, num_lags)
        :param lags: The integer-valued lag for each column of strf
        :param bias: The scalar offset

        :return: A time series y of shape (num_time_points)
    """

    input_T = np.matrix(X)
    nsamps = input_T.shape[0]
    #print 'input_T.shape=',input_T.shape
    #print 'time_lags.shape=',time_lags.shape
    #print 'filter.shape=',filter.shape

    a = np.zeros( [nsamps, 1] )
    for k,ti in enumerate(lags):
        #print '\tti=%d' % ti
        #print '\tk=%d, filter[:, k].shape=' % k,filter[:, k].shape
        at = input_T * strf[:, k].reshape(strf.shape[0], 1)
        if ti >= 0:
            if ti > 0:
                at = at[:-ti]
            #print '\tat.shape=',at.shape
            a[ti:] += at
        else:
            offset = ti % nsamps
            a[:offset] += at[-ti:]

    return a.squeeze() + bias


def make_toeplitz(input, lags, include_bias=True, fortran_style=False):
    """
        Assumes input is of dimensionality nt x nf, where nt is the number of time points and
        nf is the number of features.

        lags is an array of integers, representing the time lag from zero. a negative time lag points to the future,
        positive to the past.
    """

    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)

    if fortran_style:
        A = np.zeros([nt, nf*d+include_bias], order='F')
    else:
        A = np.zeros([nt, nf*d+include_bias])
    if include_bias:
        A[:, -1] = 1.0 # the bias term

    all_indices = np.arange(d*nf)

    #compute the channel corresponding to each parameter in the reshaped (flattened) filter
    channel_indices = np.floor(all_indices / float(d)).astype('int')

    #compute the lag index corresponding to each parameter in the reshaped (flattened) filter
    lag_indices = all_indices % d
    #print 'lag_indices=',lag_indices

    for k,i in enumerate(all_indices):
        #get lag and channel corresponding to this index
        lag_index = lag_indices[i]
        #print 'k=%d, i=%d, lag_index=%d' % (k, i, lag_index)
        lag = lags[lag_index]
        channel_to_get = channel_indices[i]

        if lag == 0:
            A[:, k] = input[:, channel_to_get]
        else:
            #shift time series for this channel up or down depending on lag
            if lag > 0:
                A[lag:, k] = input[:-lag, channel_to_get]
            else:
                A[:lag, k] = input[-lag:, channel_to_get] #note that lag is negative
    return A


def fit_strf_lasso(input, output, lags, lambda1=1.0, lambda2=1.0, num_threads=-1):
    try:
        import spams
    except ImportError:
        print('Cannot import spams! No lasso for you...')
        return

    #convert the input into a toeplitz-like matrix
    stime = time.time()
    A = make_toeplitz(input, lags, include_bias=True, fortran_style=True)
    etime = time.time() - stime
    print('[fit_strf_lasso] Time to make Toeplitz matrix: %d seconds' % etime)

    fy = np.asfortranarray(output.reshape(len(output), 1))
    #print 'fy.shape=',fy.shape
    #print 'fA.shape=',fA.shape

    #fit the STRF
    stime = time.time()
    fit_params = spams.lasso(fy, A, mode=2, lambda1=lambda1, lambda2=lambda2, numThreads=num_threads)
    etime = time.time() - stime
    print('[fit_strf_lasso] Time to fit STRF: %d seconds' % etime)

    #reshape the STRF so that it makes sense
    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)
    strf = np.array(fit_params[:-1].todense()).reshape([nf, d])
    bias = fit_params[-1].todense()[0, 0]

    return strf,bias


def fit_strf_ridge(input, output, lags, alpha=1.0, verbose=False):
    """ Fit a spatio-temporal receptive field (STRF). A STRF maps input stimuli
        into a scalar output variable. A simple ridge regression algorithm is used
        to fit the STRF.

    :param input: A numpy array of shape (num_time_points, num_spatial_channels).
    :param output: A numpy array of shape (num_time_points).
    :param lags: An array of integers that specify which time points should be included. For example,
                 to fit a STRF that uses only the stimulus information at time t to predict the output
                 at time t, specify lags=[0]. To fit a STRF that uses the stimulus information at
                 time t and the previous 10 time points, specify lags=range(0, 11).
    :param alpha: The regularization parameter for ridge regression. Try a bunch!
    :param verbose:
    :return: strf,bias: strf is a numpy array of shape (num_spatial_channels,len(lags)), which is the
            receptive field. bias is a scalar.
    """

    #convert the input into a toeplitz-like matrix
    if verbose:
        nt,nf = input.shape
        nelems = nt*nf*len(lags)
        mem = (nelems*8.) / 1024.**2
        print('[fit_strf_ridge] estimated size of toeplitz matrix: %d MB' % mem)
    stime = time.time()
    A = make_toeplitz(input, lags, include_bias=False)
    etime = time.time() - stime
    if verbose:
        print('[fit_strf_ridge] Time to make Toeplitz matrix: %d seconds' % etime)

    #fit the STRF
    stime = time.time()

    #rr = Ridge(alpha=alpha, copy_X=False, fit_intercept=True)
    rr = Ridge(alpha=alpha, fit_intercept=True)
    rr.fit(A, output)
    etime = time.time() - stime
    if verbose:
        print('[fit_strf_ridge] Time to fit STRF: %d seconds' % etime)

    #reshape the STRF so that it makes sense
    nt = input.shape[0]
    nf = input.shape[1]
    d = len(lags)
    strf = np.array(rr.coef_).reshape([nf, d])
    bias = rr.intercept_

    return strf,bias
