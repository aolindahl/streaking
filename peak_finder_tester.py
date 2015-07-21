# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:03:58 2015

@author: antlin
"""

import numpy as np
import matplotlib.pyplot as plt

import process_hdf5 as process


def max_value_position(x, y):
    return x[np.argmax(y)]

h5 = process.load_file('h5_files/run24_all.h5', verbose=2)
process.list_hdf5_content(h5)

fee_dset = h5['fee_mean']
trace_dset = h5['filtered_time_signal']
time_axis = h5['raw/time_scale'].value
peak_dset = h5['streak_peak_center']
n_events = fee_dset.shape[0]

# Plot some traces
n_traces = 16
n_col = 4
n_row = 4
# Pick some at random
idx_list = np.random.random_integers(n_events-1, size=n_traces)

plt.figure('traces')
plt.clf()
ax = None
for i, idx in enumerate(idx_list):
    ax = plt.subplot(n_row, n_col, i+1, sharex=ax)
    trace = trace_dset[i, :]
    plt.plot(time_axis, trace)

    max_value = trace.max()

    max_pos = max_value_position(time_axis, trace)
    plt.plot([max_pos]*2, [0, max_value], label='max value')

    mass_center = (time_axis * trace).sum() / trace.sum()
    plt.plot([mass_center]*2, [0, max_value], label='mass center')

    fwhm_center, _ = process.fwxm(time_axis, trace)
    plt.plot([fwhm_center]*2, [0, max_value], label='fwhm center')
    fw01m_center, _ = process.fwxm(time_axis, trace, 0.1)
    plt.plot([fw01m_center]*2, [0, max_value], label='fw0.1m center')

    sl = slice(np.searchsorted(time_axis, process.streak_time_roi[0]),
               np.searchsorted(time_axis, process.streak_time_roi[1],
                               side='right'))
    com_center = process.get_com(time_axis[sl], trace[sl])
    plt.plot([com_center]*2, [0, max_value], '--', label='com center')

    plt.plot([peak_dset[idx]]*2, [0, max_value], '-.', label='h5 center')

    plt.legend(loc='best', fontsize='x-small')

x_range = [1.59, 1.69]
#x_range = [1.5, 1.7]
ax.set_xlim(x_range)
ax.set_xticks(np.linspace(x_range[0], x_range[1], 5))
plt.tight_layout()
