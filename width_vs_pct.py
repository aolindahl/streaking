# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:56:47 2015

@author: antlin
"""
import numpy as np
import matplotlib.pyplot as plt

import process_hdf5


def select_by_distribution(data, n_sigma=1):

    try:
        for d in data:
            pass
    except:
        data = [data]

    try:
        for ns in n_sigma:
            pass
    except:
        n_sigma = [n_sigma] * len(data)

    I = np.ones_like(data[0], dtype=bool)

    for d, n_s in zip(data, n_sigma):
        mean = np.nanmean(d)
        std = np.nanstd(d)

        lower = mean - n_s * std
        upper = mean + n_s * std

        I *= (lower <= d) & (d <= upper)

    return I


if __name__ == '__main__':
    plt.ion()
    run = 102
    h5_name = process_hdf5.h5_file_name_funk(run)
    h5 = process_hdf5.load_file(h5_name)

    process_hdf5.list_hdf5_content(h5)

    raw_group = h5['raw']
    spec_group = h5['spectral_properties']

    width = spec_group['width_eV'].value
    center = spec_group['center_eV'].value
    pct = raw_group['phase_cavity_times'].value

#    I = np.ones_like(pct, dtype=bool)
#    n_sigma_pct = [0.1, 1]
#    for i in range(2):
#        I[:, i] = select_by_distribution(pct[:, i], n_sigma_pct[i])

    I = h5['pct_filter'].value
    I *= -40 < pct[:, 1]
    I *= np.isfinite(width)

    plt.figure('pct')
    plt.clf()
    plt.subplot(231)
    plt.plot(pct[I, 0], pct[I, 1], '.')
    plt.xlabel('pct 0')
    plt.ylabel('pct 1')

    h, x, y = np.histogram2d(pct[I, 0], pct[I, 1], bins=2**6)
    plt.subplot(2, 3, 4)
    plt.imshow(h.T, aspect='auto', origin='lower', interpolation='none',
               extent=(x.min(), x.max(), y.min(), y.max()))
    plt.xlabel('pct 0')
    plt.ylabel('pct 1')

    for i in range(2):
        plt.subplot(2, 3, 2+i)
        plt.plot(pct[I, i], width[I], '.')
        plt.xlabel('pct {}'.format(i))
        plt.ylabel('width')

        h, x, y = np.histogram2d(pct[I, i], width[I], bins=2**6)
        plt.subplot(2, 3, 5 + i)
        plt.imshow(h.T, aspect='auto', origin='lower', interpolation='none',
                   extent=(x.min(), x.max(), y.min(), y.max()))
        plt.xlabel('pct {}'.format(i))
        plt.ylabel('width')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        plt.savefig('figures/width_vs_phase_cavity_time_{}.{}'.format(run,
                                                                      fmt))
