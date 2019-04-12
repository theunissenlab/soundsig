from __future__ import division, print_function

import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge


class RadialBasis1D(object):
    """ Uses kernels to project elements of an array of scalars x into a higher dimensional space of
        kernel coefficients.
    """

    def __init__(self):
        self.centers = None
        self.bw = None
        self.df = None

        self.alpha = None
        self.r2 = None
        self.w = None
        self.b = None

    def fit(self, x, df=3, verbose=True, kernel='gaussian'):
        self.df = df
        # get centers of kernels, partition space into df+1 regions
        p = 100. / (df+1)
        self.centers = np.array([np.percentile(x, int((k+1)*p)) for k in range(df)])
        assert self.centers.min() >= x.min()
        assert self.centers.max() <= x.max()
        assert len(np.unique(self.centers)) == len(self.centers), '# of unique Gaussian kernels is less than the degrees of freedom!'

        # estimate bandwidth (standard deviation) for each kernel
        self.bw = np.zeros([df])
        for k in range(df):
            if k > 0:
                start = self.centers[k-1]
            else:
                start = x.min()

            if k < df-1:
                end = self.centers[k+1]
            else:
                end = x.max()

            i = (x >= start) & (x <= end)
            assert i.sum() > 1, "too few datapoints: start=%0.6f, end=%0.6f, i.sum()=%d" % (start, end, i.sum())
            # print('start=%0.6f, end=%0.6f, i.sum()=%d, bw=%0.6f' % (start, end, i.sum(), x[i].std(ddof=1)))
            self.bw[k] = x[i].std(ddof=1) / 1.

        # compute the basis representation of x
        B = self.basis(x)

        # use ridge regression to find a good representation
        alphas = np.logspace(-2, 5, 20)
        cv_sets = list(KFold(len(x), n_folds=10))

        model_perfs = list()
        for alpha in alphas:
            perfs = list()
            for k,(training_indices,test_indices) in enumerate(cv_sets):

                Xtrain = B[training_indices, :]
                ytrain = x[training_indices]

                Xtest = B[test_indices, :]
                ytest = x[test_indices]

                rr = Ridge(alpha=alpha, fit_intercept=True)
                rr.fit(Xtrain, ytrain)
                ypred = rr.predict(Xtest)

                sst = np.sum((ytest - ytrain.mean())**2)
                sse = np.sum((ytest - ypred)**2)
                r2 = 1. - (sse / sst)

                # print('Fold %d: r2=%0.2f' % (k, r2))
                perfs.append({'r2':r2, 'w':rr.coef_, 'b':rr.intercept_})

            r2_mean = np.mean([d['r2'] for d in perfs])
            w_mean = np.mean([d['w'] for d in perfs])
            b_mean = np.mean([d['b'] for d in perfs])

            model_perfs.append({'alpha':alpha, 'r2':r2_mean, 'w':w_mean, 'b':b_mean})

        # get the best model
        model_perfs.sort(key=operator.itemgetter('r2'), reverse=True)

        self.alpha = model_perfs[0]['alpha']
        self.r2 = model_perfs[0]['r2']
        self.w = model_perfs[0]['w']
        self.b = model_perfs[0]['b']

        if verbose:
            print('RadialBasis1D: alpha=%0.6f, r2=%0.2f' % (self.alpha, self.r2))
            if self.alpha == max(alphas):
                print('RadialBasis1D WARNING: maximum alpha value encountered! Something might be wrong with your data.')
                print('x on range (%0.6f, %0.6f)' % (x.min(), x.max()))
                print('centers=')
                print(self.centers)
                print('bw=')
                print(self.bw)
                self.plot(x)
                plt.show()

    def basis(self, x):
        # compute the basis
        B = np.zeros([len(x), self.df])
        for k in range(self.df):
            B[:, k] = np.exp(-(x - self.centers[k]) ** 2 / self.bw[k])
        return B


    def transform(self, x):
        # compute the basis representation for X
        B = self.basis(x)

        # multiply the basis by coefficients, elementwise
        return B * self.w

    def plot(self, x):

        xs = sorted(x)
        X = self.transform(xs)
        xsmooth = np.linspace(x.min(), x.max(), 100)

        plt.figure()
        gs = plt.GridSpec(1, 3)

        ax = plt.subplot(gs[0, :2])
        plt.hist(x, bins=20, color='m', alpha=0.7, normed=True)
        plt.plot(x, np.zeros([len(x)]), 'k|', markersize=25)
        for k in range(self.df):
            y = np.exp(-(xsmooth - self.centers[k])**2 / self.bw[k])
            plt.plot(xsmooth, y, 'b-', alpha=0.7, linewidth=4.0)

        ax = plt.subplot(gs[0, 2:])
        absmax = np.abs(X).max()
        plt.imshow(X, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, vmin=-absmax, vmax=absmax)
        plt.colorbar()


def cubic_spline_basis(x, num_knots=3, return_knots=False, knots=None):

    if knots is None:
        p = 100. / (num_knots + 1)
        knots = np.array([np.percentile(x, int((k + 1) * p)) for k in range(num_knots)])
        assert knots.min() >= x.min()
        assert knots.max() <= x.max()
        if len(np.unique(knots)) != len(knots):
            # print('[cubic_spline_basis] number of unique kernels is less than the degrees of freedom, trying wider knot spacing (q10, q50, q90)')
            knots = [np.percentile(x, 10), np.percentile(x, 50), np.percentile(x, 90)]
        assert len(np.unique(knots)) == len(knots), '# of unique kernels is less than the degrees of freedom!'

    num_knots = len(knots)
    df = num_knots+3
    B = np.zeros([len(x), df])
    for k in range(3):
        B[:, k] = x**(k+1)

    for k in range(num_knots):
        i = x > knots[k]
        B[i, k+3] = (x[i]-knots[k])**3

    if return_knots:
        return B,knots
    return B


def natural_spline_basis(x, num_knots=3):
    p = 100. / (num_knots + 1)
    knots = np.array([np.percentile(x, int((k + 1) * p)) for k in range(num_knots)])
    assert knots.min() >= x.min()
    assert knots.max() <= x.max()
    assert len(np.unique(knots)) == len(knots), '# of unique kernels is less than the degrees of freedom!'

    df = num_knots
    B = np.zeros([len(x), df])
    B[:, 0] = x

    def _dk(_k):
        _i1 = x > knots[_k]
        _i2 = x > knots[-1]
        _x1 = np.zeros([len(x)])
        _x2 = np.zeros([len(x)])

        _x1[_i1] = x - knots[_k]
        _x2[_i2] = x - knots[-1]

        _num = (_x1**3 - _x2**3)
        _denom = (knots[-1] - knots[_k])
        assert abs(_denom) > 0, "denom=0, _k=%d, _i1.sum()=%d, _i2.sum()=%d" % (_k, _i1.sum(), _i2.sum())

        return _num / _denom

    for k in range(num_knots-2):
        B[:, k + 1] = _dk(k) - _dk(num_knots-2)

    return B

    
