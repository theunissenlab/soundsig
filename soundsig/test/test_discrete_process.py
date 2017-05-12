from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt

from lasp.discrete_process import DiscreteProcessEstimator
from lasp.colormaps import magma


class DiscreteProcessTest(TestCase):


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_static_data(self):

        # specify the prior density and distribution
        ndim = 3
        pdf = np.array([0.4, 0.5, 0.1])
        cdf = np.cumsum(pdf)

        sample_rate = 200.
        duration = 60.
        dt = 1. / sample_rate
        nt = int(duration*sample_rate)

        # random event rate - events roughly every 150ms
        event_tau = 100e-3
        num_events = int(duration / event_tau)

        # generate events at each event time, drawing from the probability distribution, otherwise leave
        # the value equal to zero. zero means "no event", while an integer from 1-3 indicates an event
        # occurred
        s = np.zeros([nt], dtype='int')
        next_event = np.random.exponential(event_tau)
        t = np.arange(nt) / sample_rate
        for k,ti in enumerate(t):
            if ti >= next_event:
                # generate a sample from the distribution
                r = np.random.rand()
                try:
                    ii = np.where(cdf >= r)[0]
                    i = np.min(ii)
                except ValueError:
                    print cdf
                    print ii
                    print r
                    raise
                s[k] = i + 1
                next_event = ti + np.random.exponential(event_tau)

        print 's.unique()=',np.unique(s)

        num_events = np.sum(s > 0)
        print '# of events: %d (%0.3f Hz)' % (num_events, num_events / duration)
        p_empirical = np.zeros([ndim])
        for k in range(ndim):
            p_empirical[k] = np.sum(s == k+1)
        p_empirical /= p_empirical.sum()
        print 'True distribution: %s' % str(pdf)
        print 'Empirical distribution: %s' % str(p_empirical)

        # estimate the distribution with a variety of time constants
        taus = [100e-3, 500e-3, 1.0, 5.0]
        estimators = list()
        for tau in taus:
            est = DiscreteProcessEstimator(ndim, sample_rate, tau)
            estimators.append(est)

        # feed the estimators data and record their distributions
        P = np.zeros([len(estimators), ndim, nt])
        for ti in range(nt):
            for k,est in enumerate(estimators):
                P[k, :, ti] = est.p
                est.update(s[ti])

        # convert the signal array into a binary one-of-k matrix for visualization
        B = np.zeros([ndim, nt], dtype='bool')
        for k in range(ndim):
            i = s == k+1
            B[k, i] = True

        nrows = len(estimators) + 1
        fig = plt.figure()
        fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.35, wspace=0.35)

        plt.subplot(nrows, 1, 1)
        plt.imshow(B.astype('int'), interpolation='nearest', aspect='auto', extent=[t.min(), t.max(), 0, ndim], cmap=plt.cm.bone_r)
        plt.colorbar()

        for k in range(nrows-1):
            plt.subplot(nrows, 1, k+2)
            plt.gca().set_axis_bgcolor('black')
            plt.imshow(P[k, :, :], interpolation='nearest', aspect='auto', extent=[t.min(), t.max(), 0, ndim], cmap=magma)
            plt.colorbar()
            plt.title('tau=%0.3f' % taus[k])

        plt.show()
