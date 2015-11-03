# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:04:00 2015

@author: antlin
"""

from process_hdf5 import (load_file, list_hdf5_content, h5_file_name_funk)
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

def plot_christmas_tree(run):
    file_name = h5_file_name_funk(run)
    h5 = load_file(file_name)

    spec_group = h5['spectral_properties']
    tree_img = spec_group['histogram'].value
    center_axis_eV = spec_group['center_axis_eV'].value
    width_axis_eV = spec_group['width_axis_eV'].value

    centers_projection = np.nansum(tree_img, axis=0)
    widths_projection = np.nansum(tree_img, axis=1)

    I_width = slice(next(i for i in range(len(width_axis_eV)) if
                         widths_projection[i] > 0),
                    next(i for i in range(len(width_axis_eV), 1, -1) if
                         widths_projection[i-1] > 0))
    width_axis_eV = width_axis_eV[I_width]
    widths_projection = widths_projection[I_width]

    I_center = slice(next(i for i in range(len(center_axis_eV)) if
                          centers_projection[i] > 0),
                     next(i for i in range(len(center_axis_eV), 1, -1) if
                          centers_projection[i-1] > 0))
    center_axis_eV = center_axis_eV[I_center]
    centers_projection = centers_projection[I_center]

    tree_img = tree_img[I_width, I_center]

    center_step = np.mean(np.diff(center_axis_eV))
    center_min = min(center_axis_eV) - center_step/2
    center_max = max(center_axis_eV) + center_step/2
    width_step = np.mean(np.diff(width_axis_eV))
    width_min = min(width_axis_eV) - width_step/2
    width_max = max(width_axis_eV) + width_step/2

    plt.figure('tree ' + str(run))
    plt.clf()
    plt.subplot(221)
    plt.imshow(tree_img, aspect='auto',
               interpolation='none', origin='lower',
               extent=(center_min, center_max, width_min, width_max))
    plt.xlabel('center (eV)')
    plt.ylabel('width (eV)')

    plt.subplot(222)
    plt.plot(widths_projection, width_axis_eV)
    plt.xlabel('number of events')
    plt.ylabel('width (eV)')

    plt.subplot(223)
    plt.plot(center_axis_eV, centers_projection)
    plt.xlabel('cernter (eV)')
    plt.ylabel('number of ecents')


def plot_some_shots(run, ax=None):
    file_name = h5_file_name_funk(run)
    h5 = load_file(file_name)
    spec_group = h5['spectral_properties']

    #####
    # Select region in the tree to get traces

#    centers = (spec_group['center_eV'].value -
#               h5['photoelectron_energy_prediction_eV'].value)
    widths = spec_group['width_eV'].value

    #I = np.abs(centers) < (center_max - center_min) / 10
#    I = ((np.nanmax(widths) - (np.nanmax(widths) - np.nanmin(widths)) / 5) <
#         widths)

    I = np.isfinite(widths)

    selected = np.random.choice(np.where(I)[0], 5)
    selected.sort()

    energy_scale_eV = h5['energy_scale_eV'].value
    traces = h5['energy_signal'].value

    if ax is None:
        plt.figure('energy traces ' + str(run))
        plt.clf()
    else:
        plt.sca(ax)

    for shot in selected:
        plt.plot(energy_scale_eV, traces[shot, :] * 1e3)

    plt.xlim(70, 130)


if __name__ == '__main__':

    runs = [113, 114, 117, 108, 112, 115, 118, 109]

#    plt.figure('large width shots')
#    plt.clf()
#    ax_list = []

    for i_run, run in enumerate(runs):
        plot_some_shots(run)
        plt.ylim(-0.2, 1)
        plt.xlabel('electron energy (eV)')
        plt.ylabel('signal (arb. u.)')
        for fmt in ['pdf', 'png']:
            plt.savefig('figures/traces_{}.{}'.format(run, fmt))
