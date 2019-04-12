from __future__ import division, print_function

import copy
import numpy as np
import operator
import matplotlib.pyplot as plt

try:
    import spams
except:
    print("Cannot import spams! You won't be able to optimize!")

try:
    from cvxopt import matrix as cvxopt_matrix, solvers as cvxopt_solvers
except:
    print('Cannot import cvxopt, you won\'t be able to use the quadratic dense solver!')

import time

class SPAMSLassoSolver(object):

    def __init__(self, solver_params):

        self.lambda1 = 1.0
        if 'lambda1' in solver_params:
            self.lambda1 = solver_params['lambda1']

        self.lambda2 = 1.0
        if 'lambda2' in solver_params:
            self.lambda2 = solver_params['lambda2']

        self.positive = False
        if 'positive' in solver_params:
            self.positive = solver_params['positive']

    def solve(self, A, y, x0, as_signs):

        #print('dense_solver: y.shape=',y.shape)
        #print('dense_solver: A.shape=',A.shape)
        fy = np.asfortranarray(y.reshape(len(y), 1))
        fA = np.asfortranarray(A)
        #print('fy.shape=',fy.shape)
        #print('fA.shape=',fA.shape)
        if not self.positive:
            xnew = spams.lasso(fy, fA, mode=2, lambda1=self.lambda1, lambda2=self.lambda2)
        else:
            W = np.ones(len(x0))
            params = {'lambda1':self.lambda1, 'pos':True}
            xnew = spams.lassoWeighted(fy, fA, W, **params)

        xnew = np.array(xnew.todense()).reshape(x0.shape)

        #print('dense_solver: xnew.shape=',xnew.shape)

        return xnew

class CVXOPTQPSolver(object):
    def __init__(self, solver_params):
        self.lambda_val = 200.0
        if 'lambda' in solver_params:
            self.lambda_val = solver_params['lambda']

    def solve(self, A, y, x0, as_signs):

        print('cvxopt QP: # of params=%d' % A.shape[1])
        #flip sign of columns of A that have negative usefulness, a trick
        #Patrick Gill uses to halve the # of parameters for the QP solver
        A *= as_signs
        #the sign of x0 is unflipped for the nonzero elements, flip it
        x0 *= as_signs

        p = A.shape[1]
        H = cvxopt_matrix(np.dot(A.transpose(), A))
        f = cvxopt_matrix(self.lambda_val - np.dot(A.transpose(), y))
        b = cvxopt_matrix(np.zeros(p))
        Q = cvxopt_matrix(-np.eye(p))

        stime = time.time()
        sol = cvxopt_solvers.qp(H, f, Q, b, None, None, None, x0)
        etime = time.time() - stime
        print('QP solver took %d seconds' % int(etime))

        xnew = np.array(sol['x']).squeeze()
        #unflip elements of xnew
        xnew *= as_signs

        return xnew


class InCrowd(object):
    """ Implementation of in-crowd basis pursuit denoiser. For reference see:
        "The In-Crowd Algorithm for Fast Basis Pursuit Denoising"
        Patrick R. Gill, Albert Wang, Alyosha Molnar
        IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 59, NO. 10, OCTOBER 2011
    """

    def __init__(self, model, solver_class=SPAMSLassoSolver, solver_params=dict(), max_contrast_ratio=np.Inf, max_additions_fraction=0.75):
        self.model = model

        self.max_contrast_ratio = max_contrast_ratio
        self.max_additions_fraction = max_additions_fraction
        self.dense_solver = solver_class(solver_params)

        self.x = self.model.initial_guess()

        self.active_set = []
        self.inactive_set = np.arange(0.0, len(self.x), 1.0, dtype='int').tolist()

        self.converged = False
        self.num_iter = 0
        self.last_active_set = None
        self.active_set_signs = {}

    def iterate(self):

        print('Active set size: %d' % len(self.active_set))
        if len(self.active_set) == 0 and self.num_iter > 0:
            print('Empty active set, converging!')
            self.converged = True
            return

        if len(self.inactive_set) > 0:
            #compute usefulness
            u = self.model.usefulness(self.x, self.inactive_set)
            #normalize usefulness
            uabsmax = np.abs(u).max()
            if uabsmax > 0.0:
                u /= uabsmax

            #sort usefulness from high to low
            ulist = [(self.inactive_set[k], np.abs(uval), uval) for k,uval in enumerate(u)]
            ulist.sort(key=operator.itemgetter(1), reverse=True)
            #for (ui, uabs, u) in ulist:
            #    print( (ui, uabs, u))
            ulist = np.array(ulist)

            #get most useful elements to add to active set by thresholding using max_additions_fraction
            nzi = (ulist[:, 1] >= self.max_additions_fraction).nonzero()[0]
            #num_to_use = min(len(nzi), 25)
            num_to_use = len(nzi)
            nzi = nzi[:num_to_use]
            new_active_indices = ulist[nzi, 0].astype('int')
            if len(nzi) > 0:
                print('Adding %d active elements' % len(nzi))
            #print('new_active_indices=',new_active_indices)

            #record signs of new active set elements
            for nz in nzi:
                usign = np.sign(ulist[nz, 2])
                uindex = ulist[nz, 0]
                self.active_set_signs[uindex] = usign

            #remove newly active parameters from inactive set, add them to active set
            new_inactive_set = copy.deepcopy(self.inactive_set)

            for x_index in new_active_indices:
                new_inactive_set.remove(x_index)
                self.active_set.append(x_index)
            self.inactive_set = new_inactive_set
            #print('Active set size: %d' % len(self.active_set))

        #print('active_set=',self.active_set)
        #print('inactive_set=',self.inactive_set)

        #get submatrix of active components
        Asub = self.model.submatrix(self.active_set, include_bias=True)
        ysub = self.model.target()
        x0sub = np.concatenate([self.x[self.active_set], [self.model.bias]])

        #print('Asub.shape=',Asub.shape)
        #print('ysub.shape=',ysub.shape)
        #print('x0sub.shape=',x0sub.shape)

        #construct array of usefulness signs, used by the QP dense solver
        as_signs = []
        for asindex in self.active_set:
            as_signs.append(self.active_set_signs[asindex])
        as_signs.append(1.0)  #for the bias term

        #use interior solver to get solution for new active set
        xsub = self.dense_solver.solve(Asub, ysub, x0sub, np.array(as_signs))
        self.model.bias = xsub[-1]
        xsub = xsub[:-1]

        #print('type(xsub)=',type(xsub).__name__)

        #threshold and set new x
        self.threshold_xsub(xsub)
        #print('thresholded xsub=',xsub)
        self.x[self.active_set] = xsub

        #kick zeros out of active set
        zi = (xsub == 0.0).nonzero()[0]
        new_active_set = copy.copy(self.active_set)
        for ip in zi:
            actual_index = self.active_set[ip]
            self.inactive_set.append(actual_index)
            new_active_set.remove(actual_index)
            del self.active_set_signs[actual_index]

        if len(zi) > 0:
            print('Removing %d elements' % len(zi))

        self.active_set = new_active_set

        #check for convergence
        if self.max_additions_fraction == 0.0:
            print('Threshold is 1.0, converging.')
            self.converged = True
        else:
            if self.num_iter > 0 and len(self.active_set) == len(self.last_active_set):
                if len(np.setdiff1d(self.active_set, self.last_active_set)) == 0:
                    print('Active set has not changed, converging!')
                    self.converged = True

        self.num_iter += 1
        self.last_active_set = copy.copy(self.active_set)

    def threshold_xsub(self, xsub):
        xsubabs = np.abs(xsub)
        zero_cutoff = xsubabs.max() / self.max_contrast_ratio
        xsub[xsubabs < zero_cutoff] = 0.0


class InCrowdModel(object):
    """ Represents a problem where we want to minimize the expression
        ||y-Ax||_2 - lambda*||x||_1, where ||.||_2 and ||.||_1 are
        the L2 and L1 norms, respectively. This is an abstract class,
        actual implementations need to fill in each method.
    """

    def __init__(self):
        pass

    def initial_guess(self):
        raise Exception('initial_guess method not implemented!')

    def residual(self, x):
        raise Exception('residual method not implemented!')

    def usefulness(self, x, indices):
        """ Given a value x and a set of indices compute the "usefulness" of
            each index, i.e. which should be added to the active set (the "in crowd").
            Return a list of the same size as "indices" where each element
            has the corresponding usefulness.
        """
        raise Exception('usefulness method not implemented!')

    def submatrix(self, indices):
        """ Retrieves a submatrix of A generated by
            the columns of A indexed by "indices".
        """
        raise Exception('submatrix method not implemented!')

    def target(self):
        """ Retrieve the target output vector """
        raise Exception('target method not implemented!')



class FullInCrowdModel(InCrowdModel):
    """ This implementation uses an explicit form for the matrix A and the response vector y. """

    def __init__(self, A, y):
        """ A: a numpy.array object with the # of cols = # of params
            y: a numpy.array object with # of elements = # of data points = # of rows of A
        """
        InCrowdModel.__init__(self)

        self.A = A
        self.y = y
        self.num_params = self.A.shape[1]

    def initial_guess(self):
        return np.zeros(self.num_params)

    def residual(self, x):
        yhat = np.dot(self.A, x)
        #print('yhat=',yhat)
        r = self.y - yhat
        #print('r=',r)
        return r


    def usefulness(self, x, indices):
        """ Given a value x and a set of indices compute the "usefulness" of
            each index, i.e. which should be added to the active set (the "in crowd").
            Return a list of the same size as "indices" where each element
            has the corresponding usefulness.
        """
        r = self.residual(x)
        u = np.dot(r, self.A[:, indices])
        return u

    def submatrix(self, indices):
        """ Retrieves a submatrix of A generated by
            the columns of A indexed by "indices".
        """
        return self.A[:, indices]

    def target(self):
        return self.y


class ConvolutionalInCrowdModel(InCrowdModel):

    def __init__(self, input, target, ndelays=None, bias=0.0, group_index=None, lags=None):
        InCrowdModel.__init__(self)
        if lags is not None:
            ndelays = len(lags)
        self.ndelays = ndelays
        if lags is None:
            self.time_lags = np.arange(0, self.ndelays, 1, dtype='int')
        else:
            self.time_lags = lags
        self.input = input
        self.target_val = target
        self.num_channels = self.input.shape[1]
        self.bias = bias
        self.group_index = group_index

    def initial_guess(self):
        filter = np.zeros([self.num_channels, self.ndelays])
        return filter.reshape(self.num_channels*self.ndelays)

    def get_filter(self, x):
        """ Returns a reshaped filter based on the given vector of parameter values """
        return x.reshape(self.num_channels, self.ndelays)

    def forward(self, x):
        filter = self.get_filter(x)
        if self.group_index is None:
            yhat = fast_conv(self.input, filter, self.time_lags, bias=self.bias)
        else:
            yhat = fast_conv_grouped(self.input, filter, self.time_lags, group_index=self.group_index, bias=self.bias)

        return yhat

    def residual(self, x):
        yhat = self.forward(x)
        r = self.target_val - yhat
        return r

    def usefulness(self, x, indices):
        r = self.residual(x)
        u = np.zeros(len(indices))
        for k,i in enumerate(indices):
            a = self.submatrix([i])
            u[k] = np.dot(r, a)

        return u

    def submatrix(self, indices, include_bias=False):
        A = np.zeros([self.input.shape[0], len(indices)+include_bias])
        if include_bias:
            A[:, -1] = 1.0 # the bias term
        d = len(self.time_lags)
        m = self.num_channels
        all_indices = np.arange(d*m)

        #compute the channel corresponding to each parameter in the reshaped (flattened) filter
        channel_indices = np.floor(all_indices / float(d)).astype('int')

        #compute the lag index corresponding to each parameter in the reshaped (flattened) filter
        lag_indices = all_indices % d
        #print('lag_indices=',lag_indices)

        for k,i in enumerate(indices):
            #get lag and channel corresponding to this index
            lag_index = lag_indices[i]
            #print('k=%d, i=%d, lag_index=%d' % (k, i, lag_index))
            lag = self.time_lags[lag_index]
            channel_to_get = channel_indices[i]

            if lag == 0:
                A[:, k] = self.input[:, channel_to_get]
            else:
                #shift time series for this channel up or down depending on lag
                if lag > 0:
                    A[lag:, k] = self.input[:-lag, channel_to_get]
                else:
                    A[:lag, k] = self.input[-lag:, channel_to_get] #note that lag is negative
        return A

    def submatrix_causal(self, indices):
        A = np.zeros([self.input.shape[0], len(indices)])
        for k,i in enumerate(indices):
            #get channel are we're looking for
            channel_to_get = i % self.num_channels
            #get shift, the number of zeros that precede that channel
            downshift = np.floor(((self.ndelays*self.num_channels) - i - 0.001) / self.num_channels)
            if downshift == 0:
                A[downshift:, k] = self.input[:, channel_to_get]
            else:
                A[downshift:, k] = self.input[:-downshift, channel_to_get]
        return A

    def target(self):
        return self.target_val


def fast_conv_grouped(input, filter, time_lags, group_index, bias=0.0):
    """ Expects input to be a TxM matrix, where M = # of channels, T = # of time points """

    resp = np.zeros(input.shape[0])

    ugroups = np.unique(group_index)
    for grp in ugroups:

        gindex = (group_index == grp).nonzero()[0]
        ginput = input[gindex, :]
        resp[gindex] = fast_conv(ginput, filter, time_lags, bias=bias)

    return resp


def fast_conv(input, filter, time_lags, bias=0.0):
    """ Expects input to be a TxM matrix, where M = # of channels, T = # of time points """

    input_T = np.matrix(input)
    nsamps = input_T.shape[0]
    #print('input_T.shape=',input_T.shape)
    #print('time_lags.shape=',time_lags.shape)
    #print('filter.shape=',filter.shape)

    a = np.zeros( [nsamps, 1] )
    for k,ti in enumerate(time_lags):
        #print('\tti=%d' % ti)
        #print('\tk=%d, filter[:, k].shape=' % k,filter[:, k].shape)
        at = input_T * filter[:, k].reshape(filter.shape[0], 1)
        if ti >= 0:
            if ti > 0:
                at = at[:-ti]
            #print('\tat.shape=',at.shape)
            a[ti:] += at
        else:
            offset = ti % nsamps
            a[:offset] += at[-ti:]

    return a.squeeze() + bias
