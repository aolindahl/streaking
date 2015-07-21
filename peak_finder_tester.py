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

# Load some data [24, 26, 28, 31, 38] are all part of the tof calibration with
# the electrostatic lense on.
# The streak peak center is used in the tof calibration. So it is important
# that it contains correct values.

# Load the data
h5 = process.load_file('h5_files/run24_all.h5', verbose=2)
# Print the content
process.list_hdf5_content(h5)

# Get some important datasets.
fee_dset = h5['fee_mean']
trace_dset = h5['filtered_time_signal']
time_axis = h5['raw/time_scale'].value  # Time axis is grabbed as a value
peak_dset = h5['streak_peak_center']
n_events = fee_dset.shape[0]

# Plot some traces
n_traces = 16
# in a 4x4 grid
n_col = 4
n_row = 4
# Pick some shots at random
idx_list = np.random.random_integers(n_events-1, size=n_traces)

# Make the figure
plt.figure('traces')
plt.clf()
# Start with a None value as the axis reference
ax = None

# Iterate over the seleted shot indexes
for i, idx in enumerate(idx_list):
    # Pick the enumerated exis to plot in, share the x-axis with the previously
    # used axis (None for the first iteration)
    ax = plt.subplot(n_row, n_col, i+1, sharex=ax)

    # Get the trace of the shot
    trace = trace_dset[idx, :]
    # Plot the trace
    plt.plot(time_axis, trace)

    # Get the maximum amplitude of the trace
    max_value = trace.max()
#    # and the position in time
#    max_pos = max_value_position(time_axis, trace)
#    # Put a maeker in the plot corresponding to the maximum value
#    plt.plot([max_pos]*2, [0, max_value], label='max value')

#    # Calculate the center of mass of the complete trace
#    mass_center = (time_axis * trace).sum() / (trace.sum())
#    # and put the correspnding marker in the plot
#    plt.plot([mass_center]*2, [0, max_value], label='mass center')
#    # This actually seems to be pretty crappy. I wonder if it should?
#    # Must be sue to the finite (varying) values in the baseline.
#    # Don't use this one!! It really gets screwed up with negativa y values.

#    # Get the full with haf max center of the peak...
#    fwhm_center, _ = process.fwxm(time_axis, trace)
#    # ...and mark it in the plot
#    plt.plot([fwhm_center]*2, [0, max_value], label='fwhm center')
#    # Do the same again, but with a threshold of 0.1(max value) instead.
#    fw01m_center, _ = process.fwxm(time_axis, trace, 0.1)
#    plt.plot([fw01m_center]*2, [0, max_value], label='fw0.1m center')
#    # The difference of these two should be due to the stronger influenc of
#    # tails on the 0.1max compared to the halfmax.

    sl = slice(np.searchsorted(time_axis, process.streak_time_roi[0]),
               np.searchsorted(time_axis, process.streak_time_roi[1],
                               side='right'))
    com_center = process.get_com(time_axis[sl], trace[sl])
    plt.plot([com_center]*2, [0, max_value], '-', label='com center')

    plt.plot([peak_dset[idx]]*2, [0, max_value], '-.k', label='h5 center')

    plt.legend(loc='best', fontsize='x-small')

#x_range = [1.59, 1.69]
#x_range = [1.5, 1.7]
#ax.set_xlim(x_range)
#ax.set_xticks(np.linspace(x_range[0], x_range[1], 5))
ax.set_xlim(process.streak_time_roi)
plt.tight_layout()
