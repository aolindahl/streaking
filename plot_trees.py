# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:28:19 2015

@author: antlin
"""

import matplotlib.pyplot as plt
import numpy as np
import process_hdf5 as process

plt.ion()


def plot_tree(run, ax, title=None, draw=False):
    h5 = process.load_file(process.h5_file_name_template.format(run),
                           verbose=2)

    group = h5['spectral_properties']
    c_ax = group['center_axis_eV'][:]
    w_ax = group['width_axis_eV'][:]
    hist = group['histogram'][:]

    ax.clear()
    plt.sca(ax)
    plt.imshow(hist, aspect='auto', interpolation='none', origin='lower',
               extent=(c_ax.min(), c_ax.max(), w_ax.min(), w_ax.max()),
               vmin=0)

    plt.colorbar()

    if title is not None:
        ax.set_title(title)

    if draw:
        ax.figure.canvas.draw()

    return hist, c_ax, w_ax

if __name__ == '__main__':
    n_rows, n_cols = 2, 4
    trees = plt.figure('trees')
    trees.clear()
    trees, trees_ax_array = plt.subplots(n_rows, n_cols,
                                         sharex=True, sharey=True,
                                         num=trees.number)

    plot_tree(112, trees_ax_array[0, 0],
              'run 112: 4 fs reference, foil at -4996 um')
    plot_tree(113, trees_ax_array[1, 0],
              'run 113: 4 fs streaked')

    plot_tree(115, trees_ax_array[0, 1],
              'run 115: reference, foil at -4007 um')
    plot_tree(114, trees_ax_array[1, 1],
              'run 114: streaked')

    plot_tree(118, trees_ax_array[0, 2],
              'run 118: reference, foil at -4007 um')
    plot_tree(117, trees_ax_array[1, 2],
              'run 117: streaked')

    plot_tree(109, trees_ax_array[0, 3],
              'run 109: reference, foil out')
    plot_tree(108, trees_ax_array[1, 3],
              'run 108: streaked')

#    for ax in trees_ax_array.flatten():
#        ax.colorbar()
    for ax in trees_ax_array[:, 0]:
        ax.set_ylabel('Peak width (eV)')
    for ax in trees_ax_array[:, 1:].flatten():
        plt.setp(ax.get_yticklabels(), visible=False)

    for ax in trees_ax_array[-1, :]:
        ax.set_xlabel('Center energy (eV)')
    for ax in trees_ax_array[:-1, :].flatten():
        plt.setp(ax.get_xticklabels(), visible=False)

    trees.tight_layout()
