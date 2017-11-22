from unittest import TestCase

import numpy as np

import matplotlib.pyplot as plt

from soundsig.optimization import ConvolutionalLinearModel, finite_diff_grad, ThresholdGradientDescent
from soundsig.incrowd import fast_conv


class OptimizationTest(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_convolutional_model(self):

        nchan = 3
        nt = 1000
        nlags = 4
        X = np.random.randn(nt, nchan)
        filt = np.random.randn(nchan, nlags)
        bias = -1.5
        lags = range(nlags)

        y = fast_conv(X, filt, lags, bias)

        m = ConvolutionalLinearModel(X, y, lags=lags)

        yhat = m.forward(filt) + bias
        assert np.abs(yhat - y).sum() < 1e-6

        params = np.zeros([nchan*nlags + 1])
        params[:-1] = filt.ravel()
        params[-1] = bias
        assert m.error_vector(params).sum() < 1e-12

        #test the gradient
        params = np.random.randn(nchan*nlags + 1)
        g = m.grad(params)
        gfd = finite_diff_grad(m.error, params)
        gdiff = (g - gfd) / np.sqrt( (gfd**2).sum() )

        assert np.abs(gdiff).sum() < 1e-3

        #do some gradient descent
        m.params = np.zeros([nchan*nlags + 1])
        tgd = ThresholdGradientDescent(m, step_size=1e-3, threshold=0.0, slope_threshold=-1e-9)

        niter = 5000
        while not tgd.converged and tgd.iter < niter:
            tgd.iterate()
            print 'iter %d, err=%f' % (tgd.iter, tgd.errors[-1])

        found_bias = tgd.best_params[-1]
        found_filt = m.get_filter(tgd.best_params[:-1])

        filt_diff = (filt-found_filt)
        print 'filt=',filt
        print 'found_filt=',found_filt
        print 'filt_diff=',filt_diff
        assert np.abs(filt_diff).sum() < 1e-3

        print 'bias=',bias
        print 'found_bias=',found_bias
        print 'bias_diff=',bias-found_bias
        assert np.abs(bias-found_bias) < 1e-3
