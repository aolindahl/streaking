# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:11:23 2015

@author: antlin
"""

import matplotlib.pyplot as plt
import process_hdf5
import numpy as np

h5 = process_hdf5.load_file(process_hdf5.h5_file_name_funk(118))
process_hdf5.list_hdf5_content(h5)
raw = h5['raw']
spec = h5['spectral_properties']

# %%

fee = h5['fee_mean'].value
I = 0.004 < fee
plt.figure('fee')
plt.clf()
_, hist_lims, _ = plt.hist(fee, 2**6)
plt.hist(fee[I], hist_lims)
fee = fee[I]

# %% 
e_scale = h5['energy_scale_eV'].value
sl = slice(e_scale.searchsorted(80), e_scale.searchsorted(115))
e_scale = e_scale[sl]
e_signal = h5['energy_signal'][I, sl] * 1e3
amax = e_signal.argmax(axis=1)
sorting = np.argsort(amax)
amax_sorted = amax[sorting]
shift = - 80 - amax_sorted * np.mean(np.diff(e_scale))

e_signal = (e_signal.T / e_signal.max(axis=1)).T
e_signal_sorted = e_signal[sorting, :]
#e_pred = h5['photoelectron_energy_prediction_eV'].value[sorting]

plt.figure('traces')
plt.clf()

#img = np.zeros((len(shift)-10, np.sum((90 < e_scale) & (e_scale < 110))))
for i in range(len(shift)-10):
    plt.plot(e_scale+shift[i], e_signal_sorted[i], 'b')
#    sl = (90 < (e_scale+shift[i)]) & ((e_scale+shift[i]) < 110)
#    img[i, :] = e_signal[i, sl]

plt.figure('trace image')
plt.clf()
plt.imshow(e_signal_sorted, aspect='auto', interpolation='none', vmin=0)

#plt.xlim(-15, 15)
#plt.ylim(-0.3, 20)

# %%

width = spec['width_eV'][I]
#l3 = raw['energy_L3_MeV'][I]
#bc2 = h5['energy_BC2_MeV'][I]


plt.figure('width correlations')
plt.clf()
plt.plot(fee, width, '.')
