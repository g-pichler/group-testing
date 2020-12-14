#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import numpy as np
import scipy.spatial
from scipy.special import lambertw
from typing import Optional, Tuple


class PoolTest:
    EPSILON = 1e-12
    a: np.ndarray
    b: np.ndarray
    N: np.ndarray
    p: np.ndarray
    n_subpop: int
    n_indiv: np.ndarray

    @staticmethod
    def _binent(t: np.ndarray):
        """
        Evaluates the binary entropy function in nat.

        :param t: Numpy array of values 0 <= t <= 1
        :return: H_2(t), evaluated element-wise, in nat
        """

        assert np.all(np.logical_and(0.0 <= t, t <= 1.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            binent = -t * np.log(t) - (1 - t) * np.log(1 - t)
        binent[t == 0.0] = 0.0
        binent[t == 1.0] = 0.0
        return binent

    def _D(self, a: np.ndarray, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Evaluate \bar D(p, a, v) as defined in the reference.

        All broadcasting is applied by numpy implicitly.

        :param a: a > 0
        :param p: 0 < p < 1
        :param v: 0 <= v < 1
        :return:
        """
        assert np.all(np.logical_and(0.0 < p, p < 1.0))
        assert np.all(np.logical_and(0.0 <= v, v < 1.0))
        assert np.all(0.0 < a)

        return p * (v / (1 - v) - a * v ** a / (1 - v ** a)) + a / (1 - v ** a) - (a + v ** (a + 1)) / (
                    1 - v ** (a + 1))

    def _R(self, a: np.ndarray, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Evaluate \bar R(p, a, v) as defined in the reference.

        All broadcasting is applied by numpy implicitly.

        :param a: a > 0
        :param p: 0 < p < 1.0
        :param v: 0 <= v <=1
        :return:
        """
        assert np.all(np.logical_and(0.0 < p, p < 1.0))
        assert np.all(np.logical_and(0.0 <= v, v <= 1.0))
        assert np.all(0.0 < a)

        with np.errstate(divide='ignore', invalid='ignore'):
            R = self._D(a, p, v) * np.log(v) + self._binent(p) - np.log((1 - v ** (a + 1)) / (1 - v ** a)) + \
                p * np.log((1 - v) / (1 - v ** a))

        # We take care of NaN values arising from v == 0.0
        R_fallback = np.broadcast_to(self._binent(p), R.shape)
        R[v == 0.0] = R_fallback[v == 0.0]
        return R

    def __init__(self):
        pass

    def set(self,
            a: np.ndarray,
            p: np.ndarray,
            b: np.ndarray,
            N: np.ndarray,
            ):
        """
        Set the properties of the pooltest. This method can be called later on to change the parameters
        for an already initialized pooltest.

        For all numpy arrays, the first dimension contains the subpopulation index 0 <= i <= I-1

        The total number of subpopulations is limited to 30.

        :param a: 0 < a
        :param p: 0 < p <= 0.5
        :param b: 0 < b
        :param N: 0 < N
        """
        a = np.array(a)
        p = np.array(p)
        b = np.array(b)
        N = np.array(N)

        assert np.all(np.logical_and(0.0 < p, p <= 0.5))
        assert np.all(0.0 < a)
        assert np.all(0.0 < b)
        assert np.all(0.0 < N)

        # We put the subpopulation along the first dimension
        self.a = a.reshape((-1, 1))
        self.p = p.reshape((-1, 1))
        self.b = b.reshape((-1, 1))
        self.N = N.reshape((-1, 1))

        # Make sure that we have all parameters for all subpopulations
        assert a.shape == p.shape == b.shape == N.shape

        # Number of subpopulations
        self.n_subpop = a.shape[0]
        assert self.n_subpop > 0
        assert self.n_subpop <= 30

        # Total number of individuals
        self.n_indiv = np.sum(self.N)

    def Dmax(self) ->np.ndarray:
        """
        Compute the maximum cost, i.e., the minimal cost that can be obtained with zero tests.

        :return: [1,1] numpy array
        """
        return np.sum(self.N * self.b * np.minimum(self.a * self.p, 1 - self.p)) / self.n_indiv

    def entropy_bound(self) -> np.ndarray:
        """
        Compute average binary entropy per individual in bit.
        This is the lower bound for testing with zero cost.

        :return: [1,1] numpy array
        """
        entropy = self._binent(self.p) / np.log(2)  # we use bits
        return np.sum(self.N * entropy) / self.n_indiv

    def distortion_rate(self, n_plot: int = 1000):
        """
        Computes the distortion-rate function in bits.
        The integer n_plot specifies the nubmer of sample points.
        A  convex envelope is returned, removing spurious points due to v notin [0, v_0].

        :param n_plot: number of sample points (positive integer)
        :return: a tuple (D,R) of 1d-arrays is returned; the distortion-rate function is given in bits
        """

        # Get v for each subpopulation
        v0 = np.linspace(self.EPSILON, 1-self.EPSILON, n_plot)
        v = v0 ** self.b

        D = self._D(self.a, self.p, v)
        R = self._R(self.a, self.p, v) / np.log(2)  # We use bits

        idx1 = (1 - self.p) * v ** (self.a + 1) + self.p - v ** self.a >= 0.0
        idx2 = 1 - v - self.p * (1 - v ** (self.a + 1)) >= 0.0
        idx = np.logical_not(np.logical_and(idx1, idx2))

        D_fallback = np.broadcast_to(np.minimum(self.a * self.p, 1 - self.p), D.shape)
        D[idx] = D_fallback[idx]
        R[idx] = 0.0

        D = np.sum(self.b * self.N * D / self.n_indiv, axis=0)
        R = np.sum(R * self.N / self.n_indiv, axis=0)

        D = D.reshape((-1,))
        R = R.reshape((-1,))

        # We add the trivial points by hand
        D = np.r_[[self.Dmax()], D, [0.0]]
        R = np.r_[[0.0], R, [self.entropy_bound()]]

        # We sort the output in R, ascending
        D, R = self._cvx_hull(D, R)

        return D, R

    def _cvx_hull(self,
                  D: np.ndarray,
                  R: np.ndarray,
                  labels: Optional[np.ndarray] = None):
        """
        Computes the lower convex evelope of the function D(R).
        The output is flattened to a 1d-array and always in ascending order by rate.

        Optionally, preserves some labels for these points.

        :param D: cost
        :param R: rate
        :param labels: optional labels; shape [D.shape, ...]
        :return: return a tuple (D', R'), a subset of input values, flattened to 1D; if labels is not None,
        (D', R', labels') is returned.
        """
        assert D.shape == R.shape
        if labels is not None:
            assert labels.shape[:len(D.shape)] == D.shape
            lbl_shape = labels.shape[len(D.shape):]

        D = D.reshape((-1,))
        R = R.reshape((-1,))
        if labels is not None:
            labels = labels.reshape((-1,) + lbl_shape)

        # Combine D, R into one array
        DR = np.c_[D, R]

        # Append (Dmax, 1.0) as a catch-all on the "top-right"
        DR = np.r_['0,2,-1', DR, [self.Dmax(), 1.0]]

        # use Qhull to compute the convex hull
        hull = scipy.spatial.ConvexHull(DR)
        idx = hull.vertices

        # Remove the catch-all point again
        idx = idx[idx < DR.shape[0] - 1]

        DR = DR[idx]
        if labels is not None:
            labels = labels[idx]

        D, R = DR[:, 0], DR[:, 1]
        idx = R.argsort()
        R = R[idx]
        D = D[idx]
        if labels is not None:
            labels = labels[idx]

        r = (D, R)
        if labels is not None:
            r += (labels,)

        return r

    def _mix_points(self, D, R):
        """
        Given k values for costs and tests per individual, for each of the I subpopulations,
        this function tries all possible pairs of these k^I strategies and returns D' and R',
        each with shape [k, k, k ... k] (I dimensions in total), holding all these values.

        :param D: [I, k] numpy array
        :param R: [I, k] numpy array
        :return: (D', R') numpy arrays, each with shape [k, k ... k]
        """

        # Normalize R, D by their subpopulation sizes
        R = R * self.N / self.n_indiv
        D = D * self.N / self.n_indiv

        # Prepare large array
        shp = (D.shape[1],) * self.n_subpop
        R_ = np.zeros(shp)
        D_ = np.zeros(shp)

        for n in range(self.n_subpop):
            # We put the n-th subpopulation in position n in the array
            # and let broadcasting do the rest
            R_ += R[n, :].reshape((1,) * n + (-1,) + (1,) * (self.n_subpop - n - 1))
            D_ += D[n, :].reshape((1,) * n + (-1,) + (1,) * (self.n_subpop - n - 1))

        return D_, R_

    def SG1(self, u: Tuple[int,...] = tuple(range(100))):
        """
        Provides the distortion-rate pairs achieved by the SG1 strategy.

        The D/R pairs as well as the corresponding group sizes are returned. A convex envelope
        is computed to yield only the significant results. Output is sorted by rate in ascending
        order.

        A group size of 0 corresponds to no testing of the group, i.e., group size of infinity.

        :param u: tuple of group sizes to consider
        :return: (D, R, lbl): D and R are 1-d arrays, lbl is a [:, I] array containing the
        group sizes for each of the I subpopulations
        """
        # We use the second dimension for the group sizes
        u: np.ndarray = np.array(u, dtype=int).reshape((1, -1))

        # Compute the rate and the cost
        with np.errstate(divide='ignore'):
            R = 1 / u
        D = self.b * (1 - (1 - self.p) ** u - self.p)

        # for u == 0 we set R = 0 and D = Dmax
        idx = (u.squeeze() == 0).nonzero()[0]
        D[:, idx] = self.b * np.minimum((1 - self.p), self.a * self.p)
        R[:, idx] = 0.0

        # Mix the points to obtain all possible combinations of goup sizes
        D, R = self._mix_points(D, R)

        # We now construct the labels, using the u_label array with shape
        # [u_max, u_max, ...., u_max, I] with I+1 dimensions.
        # we prepare it, such that u_label[a_1, a_2, ... a_I, k] == a_k for k in {1,...,I} .
        # Probably there is a nicer way of doing this.
        shp = (u.shape[1],) * self.n_subpop
        u_1d = u.astype(dtype=int).squeeze()
        u_label = np.ndarray(shp + (self.n_subpop,), dtype=int)
        for n in range(self.n_subpop):
            u_label[(slice(None),) * self.n_subpop + (slice(n, n + 1),)] = u_1d.reshape(
                (1,) * n + (-1,) + (1,) * (self.n_subpop - n))

        # Disregard everything larger than Dmax
        idx = D <= self.Dmax() + self.EPSILON # add EPSILON to provide room for numerical inaccuracy
        D = D[idx]
        R = R[idx]
        u_label = u_label[idx, :]

        return self._cvx_hull(D, R, labels=u_label)

    def _SG2_tests(self, u1: np.ndarray, u2: np.ndarray):
        """
        Evaluates the necessary tests per indivudal for SG2(u1, u2).

        Special cases:
        - if u2 == 0, we have u1 == 0 and zero tests.
        - if u1 == 0, but u2 != 0 we have 1SG(u2) with 1/u2 tests per individual.

        We need u1 >= u2 or u1 == 0.

        :param u1: first group size (integer)
        :param u2: second group size (integer)
        :return:
        """

        assert np.all(np.logical_or(u1 >= u2, u1 == 0.0))

        u1_ = u1.astype(dtype=float)
        u1_[u1 == 0] = np.inf
        u2_ = u2.astype(dtype=float)
        u2_[u2 == 0] = np.inf
        with np.errstate(divide='ignore'):
            n_tests = 1 / u1_ + (1 - (1 - self.p) ** u1_) / u2_

        n_tests[np.broadcast_to(u2 == 0, n_tests.shape)] = 0.0
        return n_tests

    def SG2(self, u2: Tuple[int,...] = tuple(range(100))):
        """
        Provides the distortion-rate pairs achieved by the SG2 strategy.

        The D/R pairs as well as the corresponding group sizes are returned. A convex envelope
        is computed to yield only the significant results. Output is sorted by rate in ascending
        order.

        A group size of (0, 0) corresponds to no testing of the group, i.e., group size of infinity.
        A group size of (0, k) corresponds to a 1SG(k) strategy for this group.

        :param u2: group sizes for the second group to consider
        :return: (D, R, lbl): D and R are 1-d arrays, lbl is a [:, I, 2] array containing the two
        group sizes for each of the I subpopulations
        """

        # We use the second dimension for the group sizes
        u2: np.ndarray = np.array(u2, dtype=int).reshape((1, -1))

        # Optimitze u1 for the particular u2
        u1opt_c = np.real(2 / np.log(1 - self.p) * lambertw(-np.sqrt(-u2 * np.log(1 - self.p)) / 2))
        with np.errstate(invalid='ignore'):
            u1 = np.maximum(np.floor(u1opt_c / u2) * u2, u2)
            u1opt_hi = np.maximum(np.ceil(u1opt_c / u2) * u2, u2)

        # if u2 is zero, we don't test at all
        idx = (u2.squeeze() == 0).nonzero()[0]
        u1[:, idx] = 0
        u1opt_hi[:, idx] = 0

        idx = (self._SG2_tests(u1, u2) > self._SG2_tests(u1opt_hi, u2))
        u1[idx] = u1opt_hi[idx]
        # Check if u1 = 0 yields a better result
        idx = self._SG2_tests(u1, u2) > self._SG2_tests(np.zeros(u1.shape), u2)
        u1[idx] = 0

        # Compute Rate and Cost
        D = self.b * (1 - (1 - self.p) ** u2 - self.p)
        R = self._SG2_tests(u1, u2)

        # if u2 == 0 we set R = 0 and D = Dmax
        idx = (u2.squeeze() == 0).nonzero()[0]
        D[:, idx] = self.b * np.minimum((1 - self.p), self.a * self.p)
        R[:, idx] = 0.0

        # Mix the points to obtain all possible combinations of goup sizes
        D, R = self._mix_points(D, R)

        # We now construct the labels, using the u_label array with shape
        # [u_max, u_max, ...., u_max, I, 2] with I+1 dimensions.
        # we prepare it, such that u_label[a_1, a_2, ... a_I, k, i] == a_k for k in {1,...,I} and i == 1. For i == 0
        # it yields the optimal 1st stage group size for 2nd stage group size a_k for the k-th subpopulation.
        #
        # Probably there is also a nicer way of doing this.
        shp = (u2.shape[1],) * self.n_subpop
        u_label = np.ndarray(shp + (self.n_subpop, 2), dtype=int)
        for n in range(self.n_subpop):
            u_label[(slice(None),) * self.n_subpop + (slice(n, n + 1), slice(1, 2))] = u2.reshape(
                (1,) * n + (-1,) + (1,) * (self.n_subpop - n) + (1,))
            u_label[(slice(None),) * self.n_subpop + (slice(n, n + 1), slice(0, 1))] = u1[n,:].reshape(
                (1,) * n + (-1,) + (1,) * (self.n_subpop - n) + (1,))

        # Disregard everything larger than Dmax
        idx = D <= self.Dmax() + self.EPSILON  # add EPSILON to provide room for numerical inaccuracy
        D = D[idx]
        R = R[idx]
        u_label = u_label[idx, :, :]

        return self._cvx_hull(D, R, labels=u_label)

    def binary_splitting(self):
        """
        Provides the binary splitting upper bound from

        Aldridge, M. (2019). Rates of adaptive group testing in the linear regime.
        In: 2019 IEEE International Symposium on Information Theory (ISIT). IEEE. pp. 236–240.

        :return: tuple (D, R) with the extreme points describing the achievable
        region by binary splitting.
        """
        m = 2 ** np.floor(np.log2(1/self.p - 1))
        R = np.c_[(1 / m + (1 + np.log2(m) - 1 / m) * self.p), [0.0] * self.n_subpop]
        D = np.c_[[0.0] * self.n_subpop, self.b * np.minimum(self.a * self.p, 1 - self.p)]

        D, R = self._mix_points(D, R)
        D, R = self._cvx_hull(D, R)
        return D, R

    def individual_testing(self):
        """
        Provides the bound for individual testing.

        Each subpop is individually tested. The convex envelope is comuted.
        :return: tupe (D, R) giving the I+1 extreme points.
        """

        # We start at (1,0) and move up to (0, Dmax)
        DR = np.r_['-1,2,1', self.b * np.minimum(self.a * self.p, 1 - self.p) * self.N, -self.N] / self.n_indiv
        idx = (-DR[:, 0] / DR[:, 1]).argsort()
        DR = DR[idx, :]
        DR = np.cumsum(np.r_['0,2,1', [0.0, 1.0], DR], axis=0)

        D, R = DR[:, 0], DR[:, 1]
        idx = R.argsort()
        D, R = D[idx], R[idx]
        return D, R

