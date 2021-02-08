#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import matplotlib.pyplot as plt
from pooltesting import PoolTest
import numpy as np

class PlotHelper():
    pt: PoolTest
    n_subpop: int
    params = None
    
    def __init__(self, params):
        assert len(set([x.shape for x in params.values()])) == 1

        self.n_subpop: int = params['a'].shape[0]
        self.pt = PoolTest()
        self.fig, self.axs = plt.subplots(1, figsize=[9, 9])
        self.params = params
        self.call()

    def call(self, **kwargs):
        if kwargs:
            for i in range(self.n_subpop):
                for k in self.params.keys():
                    self.params[k][i] = kwargs[f'{k}_{i!s}']

        self.pt.set(**self.params)

        D, I = self.pt.distortion_rate()
        SG2 = self.pt.SG2(u2=range(40))
        BT = self.pt.binary_splitting()
        indiv = self.pt.individual_testing()
        #SG1 = self.pt.SG1()

        plt.clf()

        h = list()
        h.append(plt.plot(I, D, label='D(R)')[0])
        h.append(plt.plot(BT[1], BT[0], 'x-', label='Binary Splitting')[0])
        h.append(plt.plot(indiv[1], indiv[0], 'x-', label='Individual Testing')[0])
        h.append(plt.plot(SG2[1], SG2[0], 'x-', label='2SG')[0])
        h.append(plt.plot([0.0, 1.0], [self.pt.Dmax()] * 2, '--', label='D(0)')[0])
        #h.append(plt.plot(SG1[1], SG1[0], 'x-', label='1SG')[0])
        #h.append(plt.plot([self.pt.entropy_bound()] * 2, [0.0, self.pt.Dmax()], label='H_2(p)')[0])
        plt.legend(handles=h)
        plt.xlabel('Tests per individual')
        plt.ylabel('Cost per individual')
        plt.xlim((0.0, np.max(SG2[1]) * 1.1))

        #for x, y, lbl in zip(SG1[1], SG1[0], SG1[2]):
        #    plt.annotate(str(lbl), (x, y))

        for i in range(SG2[0].shape[0]):
            x = SG2[1][i]
            y = SG2[0][i]
            lbl_u1, lbl_u2 = SG2[2][i, :, 0], SG2[2][i, :, 1]
            txt = ""
            for l1, l2 in zip(lbl_u1, lbl_u2):
                txt += f'({l1},{l2}), '
            txt = txt[:-2]
            plt.annotate(txt, (x, y))

        plt.grid()
