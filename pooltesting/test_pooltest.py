from unittest import TestCase
from pooltesting import PoolTest

import numpy as np
from numpy.random import random


class TestPoolTest(TestCase):
    PLACES = 10
    EPSILON = 10**(-PLACES)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt = PoolTest()

    def test_distortion_rate(self):
        N_samples = 100
        for _ in range(100):
            for n in range(1, 5):
                a = random((n,)) * 100
                b = random((n,)) * 100
                N = random((n,)) * 1000
                p = random((n,)) * 0.5
                self.pt.set(a, p, b, N)
                D, R = self.pt.distortion_rate(N_samples)
                d0 = np.inf
                r0 = -np.inf
                for d, r in zip(D, R):
                    self.assertLessEqual(r0 - self.EPSILON, r)
                    self.assertGreaterEqual(d0 + self.EPSILON, d)
                    d0 = d
                    r0 = r

                self.assertAlmostEqual(D[0], float(self.pt.Dmax()), delta=self.EPSILON)
                self.assertAlmostEqual(R[0], 0.0, delta=self.EPSILON)
                self.assertAlmostEqual(D[-1], 0.0, delta=self.EPSILON)
                self.assertAlmostEqual(R[-1], float(self.pt.entropy_bound()), delta=self.EPSILON)

    def test__binent(self):
        p = np.array([0.5])
        self.assertTrue(np.log(2) - self.EPSILON <= self.pt._binent(p) <= np.log(2) + self.EPSILON)
        p = np.array([0.0])
        self.assertTrue(-self.EPSILON <= self.pt._binent(p) <= self.EPSILON)
        p = np.array([1.0])
        self.assertTrue(-self.EPSILON <= self.pt._binent(p) <= self.EPSILON)

    def test__cvx_hull(self):
        n = 1
        a = np.ones((n,)) * 1
        b = np.ones((n,)) * 1
        N = np.ones((n,)) * 1
        p = np.ones((n,)) * 0.1
        self.pt.set(a, p, b, N)
        D = np.array([0.01, 0.99, 0.5])
        R = np.array([0.99, 0.01, 0.5])
        labels = np.array(range(D.shape[0]))
        cvx = self.pt._cvx_hull(D, R, labels)
        self.assertTrue(cvx[0].shape[0] == 2)
        self.assertTrue(2 not in cvx[2])

    def test__D(self):
        #a = np.ones((1,)) * 1
        #p = np.ones((1,)) * 0.1
        #self.pt.set(a, p, a, a)

        # Test that D(p,a,0) == 0.0 for any p,a, and v == 0
        for a in np.linspace(self.EPSILON, 3, 30):
            a = np.array((a,))
            for p in np.linspace(self.EPSILON, 0.5-self.EPSILON, 30):
                p = np.array((p,))
                D = self.pt._D(a,p,np.zeros((1,)))
                self.assertAlmostEqual(float(D), 0.0, delta=self.EPSILON)

    def test__R(self):
        # Test that D(p,a,0) == 0.0 for any p,a, and v == 0
        for a in np.linspace(self.EPSILON, 3, 30):
            a = np.array((a,))
            for p in np.linspace(self.EPSILON, 0.5 - self.EPSILON, 30):
                p = np.array((p,))
                R = self.pt._R(a, p, np.zeros((1,)))
                self.assertAlmostEqual(float(R), self.pt._binent(p), delta=self.EPSILON)

    def test_entropy_bound(self):
        n = 1
        a = np.ones((n,)) * 1
        b = np.ones((n,)) * 1
        N = np.ones((n,)) * 1
        p = np.ones((n,)) * 0.5
        self.pt.set(a, p, b, N)
        self.assertAlmostEqual(float(self.pt.entropy_bound()), 1.0, delta=self.EPSILON)

    def test__mix_points(self):
        for _ in range(100):
            for n in range(1, 5):
                a = random((n,)) * 100
                b = random((n,)) * 100
                N = random((n,)) * 1000
                p = random((n,)) * 0.5
                self.pt.set(a, p, b, N)
                D = random((n,10))
                R = random((n, 10))
                D, R = self.pt._mix_points(D, R)
                D = np.sum(D)
                R = np.sum(R)

                D_ = np.sum(D*N)/self.pt.n_indiv
                R_ = np.sum(R*N)/self.pt.n_indiv

                self.assertAlmostEqual(float(R_), float(R), delta=self.EPSILON)
                self.assertAlmostEqual(float(D_), float(D), delta=self.EPSILON)

    def test_SG1_SG2_bounds(self):
        for _ in range(10):
            for n in range(1, 4):
                a = random((n,)) * 100
                b = random((n,)) * 100
                N = random((n,)) * 1000
                p = random((n,)) * 0.5
                self.pt.set(a, p, b, N)
                u = range(100) if n <= 3 else range(50)
                SG1 = self.pt.SG1(u=tuple(u))
                SG2 = self.pt.SG2(u2=tuple(u))
                bound = self.pt.distortion_rate(n_plot=100)
                indiv = self.pt.individual_testing()
                binsplit = self.pt.binary_splitting()

                for d, r in zip(*bound[0:2]):
                    sg2 = np.interp(r, SG2[1], SG2[0])
                    sg1 = np.interp(r, SG1[1], SG1[0])
                    idv = np.interp(r, indiv[1], indiv[0])
                    bsplt = np.interp(r, binsplit[1], binsplit[0])

                    for x in (sg2, sg1, idv, bsplt):
                        self.assertGreaterEqual(float(x), float(d) - self.EPSILON)
                    self.assertGreaterEqual(float(sg1), float(sg2) - self.EPSILON)
                    self.assertGreaterEqual(float(idv), float(sg1) - self.EPSILON)

    def test_SG2(self):
        # check that the first group size is always optimal
        u = range(100)
        for _ in range(100):
            n = 1
            a = random((n,)) * 100
            b = random((n,)) * 100
            N = random((n,)) * 1000
            p = random((n,)) * 0.5
            self.pt.set(a, p, b, N)
            SG2 = self.pt.SG2(u2=tuple(u))
            for m in SG2[2]:
                u1 = m[0,0]
                u2 = m[0,1]
                if u2 == 0:
                    # u2 == 0 implies that we do not test at all; thus also u1 == 0
                    self.assertEqual(u1, 0)
                    continue

                u2_ = np.array((u2,), dtype=int)
                val_b = np.inf
                u1_b = -1
                for mult in (0,)+tuple(range(1, u1//u2+4)):
                    u1_ = mult*u2
                    u1_t = np.array((u1_,), dtype=int)
                    val = self.pt._SG2_tests(u1_t, u2_)
                    if val <= val_b:
                        val_b = val
                        u1_b = u1_
                self.assertEqual(u1, u1_b)
