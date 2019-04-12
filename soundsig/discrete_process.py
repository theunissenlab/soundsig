from __future__ import division

import numpy as np


class DiscreteProcessEstimator(object):
    """ This class estimates the emperical distribution of a discrete random variable over time. It forgets the
        distribution with a user-specified time constant.
    """

    def __init__(self, num_states, sample_rate, time_constant):
        self.num_states = num_states
        self.state = np.zeros([num_states])
        self.sample_rate = sample_rate
        self.p = np.zeros([num_states])
        self.tau = time_constant
        self.dt = 1. / self.sample_rate
        self.eps = 1e-1

    def update(self, new_observation):
        """ Update the state of the system with a new observation.

        :param new_observation: An integer from 1 to self.num_states, indicated the category (level) of the
               variable (factor). A zero indicates that there is no observation for this time period.

        :return:
        """

        # convert the observation to one-of-k encoding
        v = np.zeros([self.num_states])
        if new_observation > 0:
            v[new_observation-1] = 1.
        self.state = self.state * (1. - (self.dt / self.tau)) + v
        self.p = (self.state + self.eps) / np.sum(self.state + self.eps)


class Discretizer(object):
    """ This class discretizes a scalar variable.
    """

    def __init__(self, xvals, num_levels=3):
        """ Initialize the class.

        :param xvals: An np.array of observations of the scalar variable.
        """
        self.num_levels = num_levels

        # choose the percentiles that separate levels
        dq = 1. / num_levels
        self.percentiles = np.zeros([num_levels-1], dtype='int')
        self.percentile_vals = np.zeros([num_levels-1])
        for k in range(num_levels-1):
            self.percentiles[k] = int((k+1)*dq*100)
            self.percentile_vals[k] = np.percentile(xvals, self.percentiles[k])

    def discretize(self, xval):
        """ Turn the scalar variable xval into the discrete category to which it belongs.

        :param xval:
        :return:
        """
        i = np.where(xval > self.percentile_vals)[0]
        if len(i) == 0:
            return 1
        else:
            return max(i) + 2
