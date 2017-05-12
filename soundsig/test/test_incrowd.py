
import numpy as np

import matplotlib.pyplot as plt

from lasp.incrowd import ConvolutionalInCrowdModel, InCrowd, fast_conv

def test2():
    #construct sample input matrix
    num_channels = 60
    num_timepoints = 10000
    stim = np.random.randn(num_channels, num_timepoints)

    #normalize stimulus so it's between -1 and 1
    stim /= np.abs(stim).max()

    #transpose matrix, incrowd expects matrix to be # of time points X # of channels
    stim = stim.transpose()

    #construct sample filter
    time_lags = np.arange(-5, 6, 1, dtype='int')
    real_filter = np.random.randn(num_channels, len(time_lags))
    #make filter sparse
    real_filter[np.abs(real_filter) < 0.90] = 0.0
    #normalize filter
    real_filter /= np.abs(real_filter).max()

    num_nonzero = (np.abs(real_filter) == 0.0).sum()
    print '# of nonzero elements in filter: %d out of %d' % (num_nonzero, len(time_lags)*num_channels)

    #construct sample output using convolution
    output = fast_conv(stim, real_filter, time_lags)

    #add random noise to output
    output += np.random.randn(len(output))*1e-6

    #create a convolutional incrowd model
    cic_model = ConvolutionalInCrowdModel(stim, output, lags=time_lags, bias=0.0)

    #create incrowd optimizer, using Lasso+Elastic net for the interior solver,
    #lambda1 is the constant for the Lasso regularization
    #lambda2 is the constant for the Elastic Net regularization
    #threshold is the fraction of parameters that are introduced into the active set at each iteration
    ico = InCrowd(cic_model, solver_params={'lambda1':1.0, 'lambda2':1.0}, max_additions_fraction=0.25)

    #run the optimization
    num_iters = 15
    for k in range(num_iters):
        if ico.converged:
            break
        ico.iterate()
        print 'Iteration %d, err=%0.9f' % (k+1, (cic_model.residual(ico.x)**2).sum())

    #get the predicted filter, make cic_model reshape the parameters into what we would expect to see
    predicted_filter = cic_model.get_filter(ico.x)
    #predicted_output = fast_conv(stim, predicted_filter, time_lags)
    predicted_output = cic_model.forward(ico.x)
    filter_diff = real_filter - predicted_filter

    plt.figure()
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    plt.imshow(real_filter, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title('Actual Filter')

    ax2 = plt.subplot2grid((2, 3), (0, 1))
    plt.imshow(predicted_filter, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title('Predicted Filter')

    ax3 = plt.subplot2grid((2, 3), (0, 2))
    plt.imshow(filter_diff, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title('Differences')

    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    plt.plot(output, 'k-', linewidth=2.0)
    plt.plot(predicted_output, 'r-', linewidth=2.0)

def test1():
    #construct sample input matrix
    num_channels = 60
    num_timepoints = 5000
    stim = np.random.randn(num_channels, num_timepoints)

    #normalize stimulus so it's between -1 and 1
    stim /= np.abs(stim).max()

    #transpose matrix, incrowd expects matrix to be # of time points X # of channels
    stim = stim.transpose()

    #construct sample filter
    time_lags = np.arange(0, 8, 1, dtype='int')
    real_filter = np.random.randn(num_channels, len(time_lags))
    #make filter sparse
    real_filter[np.abs(real_filter) < 0.90] = 0.0
    #normalize filter
    real_filter /= np.abs(real_filter).max()

    num_nonzero = (np.abs(real_filter) == 0.0).sum()
    print '# of nonzero elements in filter: %d out of %d' % (num_nonzero, len(time_lags)*num_channels)

    #construct sample output using convolution
    output = fast_conv(stim, real_filter, time_lags)

    #add random noise to output
    output += np.random.randn(len(output))*1e-6

    #create a convolutional incrowd model
    cic_model = ConvolutionalInCrowdModel(stim, output, lags=time_lags, bias=0.0)

    #create incrowd optimizer, using Lasso+Elastic net for the interior solver,
    #lambda1 is the constant for the Lasso regularization
    #lambda2 is the constant for the Elastic Net regularization
    #threshold is the fraction of parameters that are introduced into the active set at each iteration
    ico = InCrowd(cic_model, solver_params={'lambda1':1.0, 'lambda2':1.0}, max_additions_fraction=0.25)

    #run the optimization
    num_iters = 15
    for k in range(num_iters):
        if ico.converged:
            break
        ico.iterate()
        print 'Iteration %d, err=%0.9f' % (k+1, (cic_model.residual(ico.x)**2).sum())

    #get the predicted filter, make cic_model reshape the parameters into what we would expect to see
    predicted_filter = cic_model.get_filter(ico.x)
    #predicted_output = fast_conv(stim, predicted_filter, time_lags)
    predicted_output = cic_model.forward(ico.x)
    filter_diff = real_filter - predicted_filter

    plt.figure()
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    plt.imshow(real_filter, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title('Actual Filter')

    ax2 = plt.subplot2grid((2, 3), (0, 1))
    plt.imshow(predicted_filter, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title('Predicted Filter')

    ax3 = plt.subplot2grid((2, 3), (0, 2))
    plt.imshow(filter_diff, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.title('Differences')

    ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    plt.plot(output, 'k-', linewidth=2.0)
    plt.plot(predicted_output, 'r-', linewidth=2.0)


def test_conv_single():
    test_conv(N=1000, M=1)


def test_conv(N=1000, M=10):

    #test causal convolution
    delays = np.arange(0, 20, 1, dtype='int')
    D = len(delays)
    A = np.random.randn(N, M)
    w = np.random.randn(M, D)
    yslow = slow_conv(A, w, delays)
    #print yslow
    yfast = fast_conv(A, w, delays)
    #print yfast
    ydiff = yslow - yfast
    #print list(ydiff)

    total_diff = np.abs(ydiff).sum()
    print 'total diff for causal:%f' % total_diff

    assert total_diff < 1e-8

    #test acausal convolution
    delays = np.arange(-10, 11, 1, dtype='int')
    D = len(delays)
    A = np.random.randn(N, M)
    w = np.random.randn(M, D)
    yslow = slow_conv(A, w, delays)
    #print yslow
    yfast = fast_conv(A, w, delays)
    #print yfast
    ydiff = yslow - yfast
    #print list(ydiff)

    total_diff = np.abs(ydiff).sum()
    print 'total diff for acausal:%f' % total_diff

    assert total_diff < 1e-8

    #test non-contiguous convolution
    delays = np.arange(-20, 22, 2, dtype='int')
    D = len(delays)
    A = np.random.randn(N, M)
    w = np.random.randn(M, D)
    yslow = slow_conv(A, w, delays)
    #print yslow
    yfast = fast_conv(A, w, delays)
    #print yfast
    ydiff = yslow - yfast
    #print list(ydiff)

    total_diff = np.abs(ydiff).sum()
    print 'total diff for non-contiguous:%f' % total_diff

    assert total_diff < 1e-8


def slow_conv(A, w, delays):
    """ A simple implementation of convolution to compare against fast_conv """

    #A is a NxM matrix, N is number of samples, M is dimensionality
    Mw,D = w.shape
    N,M = A.shape
    assert Mw == M

    y = np.zeros([N])

    for k in range(N):
        x = np.zeros([M, D])
        for m,ti in enumerate(delays):
            Aindex = k - ti
            if Aindex >= 0 and Aindex < N:
                x[:, m] = A[Aindex, :]
        y[k] = (x*w).sum()

    return y
