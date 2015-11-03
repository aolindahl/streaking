# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:13:02 2015

@author: antlin
"""

import process_hdf5
import matplotlib.pyplot as plt
import numpy as np
import aolPyModules
import lmfit

run = 98
h5 = process_hdf5.load_file(process_hdf5.h5_file_name_funk(run), plot=True)

process_hdf5.list_hdf5_content(h5)

raw = h5['raw']
spectral = h5['spectral_properties']

# Plot the phase cavity times
pct = raw['phase_cavity_times']
plt.figure('Phase cavity times')
plt.clf()
pct_selection = (np.isfinite(np.sum(pct, axis=1)) &
                 (0.1 < pct[:, 0]) & (pct[:, 0] < 2) &
                 (-2 < pct[:, 1]) & (pct[:, 1] < 2))
#                 (pct[:, 0] > -50) & (pct[:, 0] < 50))
#pct_selection = h5['pct_filter'].value

pct = pct[pct_selection, :]

for i in range(2):
    plt.subplot(1, 3, i+1)
    plt.title('Time {}'.format(i))
    hist, hist_edges = np.histogram(pct[:, i], bins=100)
    plt.bar(hist_edges[: -1], hist, width=np.diff(hist_edges))

plt.subplot(133)
plt.plot(pct[:, 0], pct[:, 1], '.')


def residuals(params, x, y=None):
    a = params['amplitude'].value
    c = params['center'].value
    s = params['sigma'].value
    b = params['background'].value
    k = params['slope'].value

    mod = lmfit.models.gaussian(x, a, c, s) + b + k * x

    if y is None:
        return mod
    return mod-y


def start_params(x=None, y=None):
    params = lmfit.Parameters()
    params.add('amplitude', 0.1, min=0)
    params.add('center',
               (3.2 if x is None else np.nanmean(x)),
               min=(3 if x is None else np.nanmin(x)),
               max=(3.5 if x is None else np.nanmax(x)))
    params.add('sigma', 0.2, min=0, max=0.5)
    params.add('background',
               (15 if y is None else np.nanmean(y)),
               min=0, max=(20 if y is None else np.nanmax(y)))
    params.add('slope', 0)
    return params


plt.figure('delay scan run {}'.format(run))
plt.clf()

t0 = 1.3477e4

fs_time = raw['fs_angle_shift'][pct_selection] / 1000 - t0
#dither = (np.random.rand(len(fs_time)) - 0.5) * 0.05
dither = 0
center = spectral['center_eV'][pct_selection]
width = spectral['width_eV'][pct_selection]
event_time = raw['event_time_s'][pct_selection]

#t_min, t_max = 2.1, 4.1
#t_min, t_max = fs_time.min()-0.5, 30
t_min, t_max = 5, 27
w_min, w_max = 0, 15
c_min, c_max = 90, 105
time_ax = np.linspace(t_min, t_max, 2**8+1)[1::2]
width_ax = np.linspace(w_min, w_max, 2**7+1)[1::2]
center_ax = np.linspace(c_min, c_max, 2**7+1)[1::2]

dw = np.mean(np.diff(width_ax))
dt = np.mean(np.diff(time_ax))*1e3
dc = np.mean(np.diff(center_ax))

time_w_dither = fs_time+dither
time_w_pct = fs_time-pct[:, 1]

t_selection = (t_min < time_w_pct) & (time_w_pct < t_max)

params1 = start_params(time_w_dither[t_selection], width[t_selection])
lmfit.minimize(residuals, params1, args=(time_w_dither[t_selection],
                                         width[t_selection]))

params2 = start_params(time_w_pct[t_selection], width[t_selection])
lmfit.minimize(residuals, params2, args=(time_w_pct[t_selection],
                                         width[t_selection]))

params_center = start_params(time_w_pct[t_selection], center[t_selection])
params_center['background'].value = np.nanmean(width)
params_center['background'].min = 0
params_center['background'].max = 110
params_center['slope'].value = 0
lmfit.minimize(residuals, params_center, args=(time_w_pct[t_selection],
                                               center[t_selection]))

print '\nparams1:'
lmfit.report_errors(params1)
print '\nparams2:'
lmfit.report_errors(params2)
print '\nparams_center:'
lmfit.report_errors(params_center)

img1 = aolPyModules.plotting.center_histogram_2d(time_w_dither, width,
                                                 time_ax, width_ax)
img2 = aolPyModules.plotting.center_histogram_2d(time_w_pct, width,
                                                 time_ax, width_ax)
img_center = aolPyModules.plotting.center_histogram_2d(time_w_pct, center,
                                                       time_ax, center_ax)

mean1 = img1.T.dot(width_ax) / img1.sum(0)
mean2 = img2.T.dot(width_ax) / img2.sum(0)
mean_center = img_center.T.dot(center_ax) / img_center.sum(0)

deviation_mat = (
    np.repeat(center_ax.reshape(1, -1), len(mean_center), axis=0) -
    np.repeat(mean_center.reshape(-1, 1), len(center_ax), axis=1))**2
std_center = np.sqrt((img_center.T * deviation_mat).sum(axis=1) /
                     img_center.sum(0))

imshow_kw = {'aspect': 'auto', 'origin': 'low', 'interpolation': 'none',
             'extent': (t_min, t_max, w_min, w_max)}

ax1 = plt.subplot(221)
plt.plot(time_w_dither, width, '.')
plt.plot(time_ax, mean1, 'y', linewidth=3,
         label='mean width in {:.1f} fs bins'.format(dt))
sigma1 = params1['sigma'].value
fwhm1 = 2 * np.sqrt(2 * np.log(2)) * sigma1
plt.plot(time_ax, residuals(params1, time_ax), 'r', linewidth=3,
         label=('gaussian fit, sigma={:.0f} fs'.format(1e3*sigma1) +
                ', fwhm={:.0f} fs'.format(1e3*fwhm1)))
plt.ylabel('spectral width (eV)')
plt.xlabel('angle shift time (ps)')
plt.legend(loc='best')

plt.subplot(223, sharex=ax1, sharey=ax1)
plt.plot(time_w_pct, width, '.')
plt.plot(time_ax, mean2, 'y', linewidth=3,
         label='mean width in {:.1f} fs bins'.format(dt))
sigma = params2['sigma'].value
fwhm1 = 2 * np.sqrt(2 * np.log(2)) * sigma
plt.plot(time_ax, residuals(params2, time_ax), 'r', linewidth=3,
         label=('gaussian fit, sigma={:.0f} fs'.format(1e3*sigma) +
                ', fwhm={:.0f} fs'.format(1e3*fwhm1)))
plt.legend(loc='best')
plt.xlabel('angle shift time - phse cavity time (ps)')
plt.ylabel('spectral width (eV)')
plt.legend(loc='best')

plt.subplot(222, sharex=ax1, sharey=ax1)
plt.imshow(img1, **imshow_kw)
plt.plot(time_ax, residuals(params1, time_ax), 'r', linewidth=3,
         label=('gaussian fit, sigma={:.0f} fs'.format(1e3*sigma1) +
                ', fwhm={:.0f} fs'.format(1e3*fwhm1)))
plt.xlabel('angle shift time (ps)')
plt.ylabel('spectral width (eV)')
plt.legend(loc='best')

plt.subplot(224, sharex=ax1, sharey=ax1)
plt.imshow(img2, **imshow_kw)
plt.plot(time_ax, residuals(params2, time_ax), 'r', linewidth=3,
         label=('gaussian fit, sigma={:.0f} fs'.format(1e3*sigma) +
                ', fwhm={:.0f} fs'.format(1e3*fwhm1)))
plt.xlabel('angle shift time - phse cavity time (ps)')
plt.ylabel('spectral width (eV)')
plt.legend(loc='best')

plt.axis(imshow_kw['extent'])
plt.tight_layout()
plt.savefig('figures/delay_scan_{}.pdf'.format(run))


#plt.figure('deviation')
#plt.clf()
#
#ax5 = plt.subplot(211, sharex=ax1)
#plt.plot(time_w_pct, center - mean_center, '.')
##plt.plot(time_ax, mean_center, 'y', linewidth=3,
##         label='mean center in {:.1f} fs bins'.format(dt))
#plt.plot(time_ax, std_center, 'm', linewidth=3,
#         label='std center in {:.1f} fs bins'.format(dt))
##sigma_c = params_center['sigma'].value
##fwhm_c = 2 * np.sqrt(2 * np.log(2)) * sigma_c
##plt.plot(time_ax, residuals(params_center, time_ax), 'r', linewidth=3,
##         label=('gaussian fit, sigma={:.0f} fs'.format(1e3*sigma_c) +
##                ', fwhm={:.0f} fs'.format(1e3*fwhm_c)))
#plt.xlabel('angle shift time - phse cavity time (ps)')
#plt.ylabel('spectral center (eV)')
#plt.legend(loc='best')
#
#plt.subplot(212, sharex=ax1, sharey=ax5)
#imshow_kw['extent'] = (t_min, t_max, c_min, c_max)
#plt.imshow(img_center, **imshow_kw)
#plt.plot(time_ax, residuals(params_center, time_ax), 'r', linewidth=3,
#         label=('gaussian fit, sigma={:.0f} fs'.format(1e3*sigma_c) +
#                ', fwhm={:.0f} fs'.format(1e3*fwhm_c)))
#plt.xlabel('angle shift time - phse cavity time (ps)')
#plt.ylabel('spectral center (eV)')
#plt.legend(loc='best')

plt.figure('pct drift')
plt.clf()
plt.plot(event_time, pct[:, 1], '.')
#plt.plot(event_time[:-1], np.diff(event_time), '.')
