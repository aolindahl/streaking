# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:32:27 2015

@author: antlin
"""

from process_hdf5 import (load_file, list_hdf5_content)
import matplotlib.pyplot as plt

plt.ion()


file_name = 'h5_files/run108_all.h5'
file_name = 'h5_files/run28_all.h5'

h5 = load_file(file_name, verbose=True, plot=True)

list_hdf5_content(h5)
raw_group = h5['raw']
spectral_group = h5['spectral_properties']

n_row = 3
n_col = 4

fee = h5['fee_mean'][:] * 1e3
l3_energy = raw_group['energy_L3_MeV'][:]
bc2_energy = h5['energy_BC2_MeV'][:]
tof = h5['streak_peak_center'][:]

I = fee > 1
if '108' in file_name:
    I *= (4610 < l3_energy) & (l3_energy < 4650)
    I *= (4980 < bc2_energy) & (bc2_energy< 5020)
    I *= 50 < fee
    I *= (1.606 < tof) & (tof < 1.614)
elif '28' in file_name:
    I *= (1.606 < tof) & (tof < 1.612)
    

fee = fee[I]
l3_energy = l3_energy[I]
bc2_energy = bc2_energy[I]
tof = tof[I]
e_beam_energy_diff = l3_energy - bc2_energy
energy_center = spectral_group['gaussian_center'][I]

l3_energy -= l3_energy.mean()
bc2_energy -= bc2_energy.mean()
e_beam_energy_diff -= e_beam_energy_diff.mean()
tof -= tof.mean()
tof *= 1e3

def plot_stuff(x, y, c, ax=None, xlabel=None, ylabel=None, clabel=None):
    if ax is not None:
        plt.sca(ax)

    plt.scatter(x, y, s=2, c=c, linewidths=(0,))

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if clabel is not None:
        cbar = plt.colorbar()
        cbar.set_label(clabel)

fig = plt.figure('Correlations')
plt.clf()
fig, ax_list = plt.subplots(n_row, n_col, num=fig.get_label())
ax_list = ax_list.flatten()

l3_label = 'L3 energy deviation (MeV)'
bc2_label = 'BC2 energy deviation (MeV)'
fee_label = u'pulse energy (µJ)'
tof_label = 'tof_deviation (ns)'
diff_label = 'L3 - BC2 diff deviation (MeV)'

i_ax = 0
plot_stuff(l3_energy, fee, fee, ax_list[i_ax],
           l3_label, fee_label, fee_label)

i_ax += 1
plot_stuff(bc2_energy, fee, fee, ax_list[i_ax],
           bc2_label, fee_label, fee_label)

i_ax += 1
plot_stuff(e_beam_energy_diff, fee, fee, ax_list[i_ax],
           diff_label, fee_label, fee_label)

i_ax += 1
plot_stuff(l3_energy, bc2_energy, fee, ax_list[i_ax],
           l3_label,bc2_label, fee_label)

i_ax += 1
plot_stuff(l3_energy, tof, fee, ax_list[i_ax],
           l3_label, tof_label, fee_label)

i_ax += 1
plot_stuff(bc2_energy, tof, fee, ax_list[i_ax],
           bc2_label, tof_label, fee_label)

i_ax += 1
plot_stuff(e_beam_energy_diff, tof, fee, ax_list[i_ax],
           diff_label, tof_label, fee_label)

i_ax += 1
plot_stuff(l3_energy, bc2_energy, tof, ax_list[i_ax],
           l3_label, bc2_label, tof_label)

i_ax += 1
plot_stuff(tof, fee, fee, ax_list[i_ax],
           tof_label, fee_label, fee_label)

i_ax += 1
plot_stuff(tof, e_beam_energy_diff, fee, ax_list[i_ax],
           tof_label, diff_label, fee_label)


plt.tight_layout()
