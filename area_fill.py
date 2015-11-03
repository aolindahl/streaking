# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 08:55:06 2015

@author: antlin
"""

import numpy as np
import process_hdf5
import matplotlib.pyplot as plt


def zero_crossing_area(y):
    # Find zero crossings around peak
    i_max = np.argmax(y)

    i_end = i_max + np.argmax(y[i_max:] < 0)
    i_start = i_max - np.argmax(y[i_max:: -1] < 0) + 1

    if (i_end <= i_max) or (i_start > i_max):
        return np.nan, [np.nan, np.nan]

    return y[i_start: i_end].sum(), [i_start, i_end]


if __name__ == '__main__':
    plt.ion()

    # get a file
    h5 = process_hdf5.load_file(process_hdf5.h5_file_name_funk(101))

    # list the content
    process_hdf5.list_hdf5_content(h5)

    # get the energy scale
    e_axis_eV_full = h5['energy_scale_eV'].value
#    e_slice = slice(np.searchsorted(e_axis_eV_full, 65),
#                    np.searchsorted(e_axis_eV_full, 150))
#    e_slice = slice(None)
    e_slice = slice(np.searchsorted(e_axis_eV_full, 50), None)
    e_axis_eV = e_axis_eV_full[e_slice]
    de = np.mean(np.diff(e_axis_eV_full))

    # pick a trace
    trace = h5['energy_signal'][
        np.random.randint(h5['energy_signal'].shape[0]), e_slice]

    fig = plt.figure('trace')
    plt.clf()
    plt.plot(e_axis_eV, trace)
    plt.plot(e_axis_eV, np.zeros_like(e_axis_eV), '--k')

    # Find zero crossings around peak
    A_peak, [i_start, i_end] = zero_crossing_area(trace)
    A_peak *= 1.1

    plt.plot(e_axis_eV[i_start: i_end], trace[i_start: i_end], 'og')
    fig.canvas.draw()

    i_order = np.argsort(trace).tolist()[::-1]

    temp_start = i_start = i_order[0]
    temp_end = i_end = i_order[0]

    def area_check(y, start, end, A):
        selection = y[start: end+1]
        a = (selection * (selection > 0)).sum()
        return a >= A

    A = 0
    i_last = len(trace) - 1
    exit_flag = False
    i_list = []
    for i_num, i, in enumerate(i_order):
        if (i_start <= i) and (i <= i_end):
            continue

        if (i_start - 1 > i) or (i_end + 1 < i):
            i_list.append(i)

    #    plt.plot(e_axis_eV[i_start: i_end], trace[i_start: i_end], '.r')
    #    plt.plot(e_axis_eV[i], trace[i], 'c^')
    #    fig.canvas.draw()
    #    raw_input('enter...')

        while i < i_start:
            i_start -= 1
            if (trace[i_start] > trace[i_start + 1]):
                while (((i_end < i_last) and
                        (trace[i_end + 1] < 0) and (trace[i_end] < 0)) and
                       (trace[i_end - 1] < trace[i_start])):
                    i_end -= 1
            else:
                while (((trace[i_start] < 0) and (trace[i_end] > 0)) and
                       (i_end < i_last) and (trace[i_end] > trace[i_start])):
                    i_end += 1

            if area_check(trace, i_start, i_end, A_peak):
                exit_flag = True
                break

        while i > i_end:
            i_end += 1
            if (trace[i_end] > trace[i_end - 1]):
                while (((i_start > 0) and
                        (trace[i_start - 1] < 0) and (trace[i_start] < 0)) and
                       (trace[i_start + 1] < trace[i_end])):
                    i_start += 1
            else:
                while (((trace[i_end] < 0) and (trace[i_start] > 0)) and
                       (i_start > 0) and (trace[i_start] > trace[i_end])):
                    i_start -= 1

            if area_check(trace, i_start, i_end, A_peak):
                exit_flag = True
                break

        if exit_flag:
            while trace[i_start] < 0:
                i_start += 1
            while trace[i_end] < 0:
                i_end -= 1
            break

    plt.plot(e_axis_eV[i_start: i_end+1], trace[i_start: i_end+1], '.r')
    plt.plot(e_axis_eV[i_list], trace[i_list], 'co', markersize=15,
             markerfacecolor='none', mec='c', mew=2)
