# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:37:51 2015

@author: Anton O Lindahl
"""
import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import lmfit

from aolPyModules import wiener, wavelet_filter
import time_to_energy_conversion as tof_to_energy
from aolPyModules import plotting as aol_plotting

prompt_roi = [1.508, 1.535]
streak_time_roi = [1.57, 1.66]
wt_th = 0.03
energy_scale_eV = np.linspace(40, 160, 2**9)
time_stamp = 'time_stamp'
data_dir = 'h5_files'
h5_file_name_template = data_dir + '/run{}_all.h5'
response_file_name = data_dir + '/response.h5'
nois_file_name = data_dir + '/noise.h5'
tof_to_energy_conversion_file_name = data_dir + '/time_to_energy.h5'


def h5_file_name_funk(run):
    return h5_file_name_template.format(run)


def update_progress(i_evt, n_events, verbose=True):
    if (verbose and
            ((i_evt % (n_events / 100) == 0) or (i_evt == n_events-1))):
        progress = (100 * i_evt) / (n_events - 1)
        num_squares = 40
        base_string = '\r[{:' + str(num_squares) + '}] {}%'
        print base_string.format('#'*(progress * num_squares / 100), progress),
        sys.stdout.flush()


def list_hdf5_content(group, indent='  '):
    for k, v in group.iteritems():
        print '{}"{}"'.format(indent, k),
        if isinstance(v, h5py.Group):
            print 'group with members:'
            list_hdf5_content(v, indent=indent + '  ')
        elif isinstance(v, h5py.Dataset):
            print '\t{} {}'.format(v.shape, v.dtype)


def make_dataset(h5, name, shape, dtype=np.float):
    try:
        dset = h5.require_dataset(name, shape=shape,
                                  dtype=dtype, exact=True)
    except TypeError:
        del h5[name]
        dset = h5.create_dataset(name, shape=shape, dtype=np.float)
    if time_stamp not in dset.attrs.keys():
        dset.attrs.create(time_stamp, 0)
    return dset


def make_group(h5, name):
    try:
        group = h5.require_group(name)
    except TypeError:
        del h5[name]
        group = h5.create_group(name)
    if time_stamp not in group.attrs.keys():
        group.attrs.create(time_stamp, 0)
    return group


def older(dset, dset_list):
    if (isinstance(dset_list, h5py.Dataset) or
            isinstance(dset_list, h5py.Group)):
        return dset.attrs[time_stamp] < dset_list.attrs[time_stamp]

    return np.any([dset.attrs[time_stamp] < d.attrs[time_stamp] for
                   d in dset_list])


class Timer_object:
    def __init__(self, t):
        self.attrs = {'time_stamp': t}


def get_response(plot=False, verbose=0):
    try:
        with h5py.File(response_file_name, 'r') as f:
            response = f['signal'].value
            t = f['signal'].attrs[time_stamp]
    except IOError:
        if verbose > 0:
            print 'Could not open response file. Trying to make it.'
        response, t = construct_response(verbose=verbose)
    if plot:
        with h5py.File(response_file_name, 'r') as f:
            time_scale = f['time_scale'].value
        plt.figure('response')
        plt.clf()
        plt.plot(time_scale, response)

    return response, t


def construct_response(plot=False, verbose=0):
    # The Kr runs
    runs = [132, 133, 134, 135, 136]
    if verbose > 0:
        print 'Loading Kr files for prompt determination.'
    h5_file_names = [h5_file_name_template.format(run) for run in runs]
    h5_list = []
    for file_name in h5_file_names:
        update_run_contained_derived_data(file_name, verbose=verbose)
        h5_list.append(h5py.File(file_name, 'r+'))
    time_scale = h5_list[0]['raw/time_scale'].value
    response = np.zeros_like(time_scale)
    n_shots = 0
    sl = slice(time_scale.searchsorted(prompt_roi[0]),
               time_scale.searchsorted(prompt_roi[1], side='right'))
    for h5 in h5_list:
        response[sl] += h5['raw/time_signal'][:, sl].sum(0)
        n_shots += h5['raw/event_time_s'].shape[0]
    response /= n_shots

    response[sl] = wiener.edgeSmoothing(response[sl], smoothPoints=15)

    response /= response.sum()

    with h5py.File(response_file_name, 'w') as res_file:
        dset = res_file.create_dataset('signal', data=response)
        dset.attrs.create(time_stamp, time.time())
        res_file.create_dataset('time_scale', data=time_scale)

    return get_response(plot=plot, verbose=verbose)


def get_file_names_for_noise_spectrum():
    return ['/'.join([data_dir, f]) for f in os.listdir(data_dir) if
            f.startswith('run') and f.endswith('_all.h5')]


def get_nois_spectrum(plot=False, verbose=0):
    try:
        with h5py.File(nois_file_name, 'r') as f:
            pass
        new_noise = False
    except IOError:
        if verbose > 0:
            print 'Could not open response file. Trying to make it.',
            print 'In "get_nois_spectrum()".'
        construct_nois_spectrum(plot=plot, verbose=verbose)
        new_noise = True

    if not new_noise:
        make_new_noise = False
        with h5py.File(nois_file_name, 'r') as f:
            noise = f['noise']
            h5_file_names = get_file_names_for_noise_spectrum()
            for h5_name in h5_file_names:
                with h5py.File(h5_name, 'r') as h5:
                    if older(noise, h5['raw']):
                        make_new_noise = True
                        if verbose > 0:
                            print 'Noise was made earlier than the raw data',
                            print 'in the file', h5_name, 'Make new noise.'
                        break
                    elif False:
                        print 'Noise was made later than the raw data in',
                        print 'the file', h5_name

        if make_new_noise:
            construct_nois_spectrum(plot=plot, verbose=verbose)

    with h5py.File(nois_file_name, 'r') as f:
        noise = f['noise']
        return noise.value, noise.attrs['time_stamp']


def construct_nois_spectrum(plot=False, verbose=0):
    h5_file_names = get_file_names_for_noise_spectrum()
    for file_name in h5_file_names:
        update_run_contained_derived_data(file_name)
    empty_shots = []
    for i, h5_name in enumerate(h5_file_names):
        with h5py.File(h5_name, 'r') as h5:
            time_signal_dset = h5['raw/time_signal']
            try:
                max_signal = h5['max_signal'].value
            except KeyError:
                max_signal = np.max(time_signal_dset.value, axis=1)
            no_x_rays = max_signal < 0.04
            if no_x_rays.sum() > 0:
                empty_shots.extend(time_signal_dset[no_x_rays, :])
            if i == 0:
                time_scale = h5['raw/time_scale'].value
            if verbose > 0:
                print h5_name, 'has', no_x_rays.sum(), 'empty shots'
    empty_shots = np.array(empty_shots)

#    print len(empty_shots)
#    plt.figure('snr')
#    plt.clf()
#    for shot in empty_shots[:]:
#        plt.plot(time_scale, shot)

    freq = (np.linspace(0., 1., len(time_scale)) *
            1e-3/(time_scale[1] - time_scale[0]))
    fft_empty_shots = np.fft.fft(empty_shots, axis=1)
    amp = np.mean(np.abs(fft_empty_shots)**2, axis=0)
    wt_amp = amp[:]
    wt_amp = wavelet_filter.wavelet_filt(amp[1:], thresh=wt_th)
    wt_amp[1:] = (wt_amp[1:] + wt_amp[-1:0:-1]) / 2

#    plt.figure('fft')
#    plt.clf()
#    plt.plot(freq, amp)
#    plt.plot(freq, wt_amp, 'r')

    with h5py.File(nois_file_name, 'w') as f:
        dset = f.create_dataset('noise', data=wt_amp)
        dset.attrs.create('time_stamp', time.time())
        f.create_dataset('freq', data=freq)

    return get_nois_spectrum()


def construct_snr_spectrum(h5, plot=False):
    noise, t = get_nois_spectrum()

    sig_spec = h5['fft_spectrum_mean'].value
    freq = h5['fft_freq_axis'].value

    wt_spec = wavelet_filter.wavelet_filt(sig_spec, thresh=wt_th)
    wt_spec[1:] = (wt_spec[1:] + wt_spec[-1:0:-1]) / 2

    snr = (wt_spec - noise) / noise

    if plot:
        plt.figure('signal and noise')
        plt.clf()
        plt.semilogy(freq, sig_spec, label='signal')
        plt.semilogy(freq, noise, label='noise')
        plt.semilogy(freq, wt_spec, label='wt signal')
        plt.semilogy(freq, snr, label='snr')
        plt.legend(loc='best')

    return snr


def check_tof_to_energy_conversion_matrix(plot=False, verbose=0):
    try:
        with h5py.File(tof_to_energy_conversion_file_name, 'r'):
            pass
    except IOError:
        if verbose > 0:
            print 'Could not open the file. Making the conversion matrix.'
        construc_tof_to_energy_conversion_matrix(plot=plot, verbose=verbose)

    _, h5_dict, _ = tof_to_energy.load_tof_to_energy_data(verbose=verbose)
    with h5py.File(tof_to_energy_conversion_file_name, 'r') as trans_h5:
        if not older(
                trans_h5['matrix'],
                [h5['streak_peak_integral'] for h5 in h5_dict.itervalues()] +
                [Timer_object(1437117486)]):
            return
    if verbose > 0:
        print 'Conversion to old, remaking it.'
    construc_tof_to_energy_conversion_matrix(plot=plot, verbose=verbose)


def construc_tof_to_energy_conversion_matrix(plot=False, verbose=0):
    M, t, E, time_to_energy_params, tof_prediction_params = \
        tof_to_energy.make_tof_to_energy_matrix(
            energy_scale_eV=energy_scale_eV, plot=plot, verbose=verbose)
    with h5py.File(tof_to_energy_conversion_file_name, 'w') as h5:
        dset = h5.create_dataset('matrix', data=M)
        dset.attrs.create('time_stamp', time.time())
        dset = h5.create_dataset('time_scale', data=t)
        dset.attrs.create('time_stamp', time.time())
        dset = h5.create_dataset('energy_scale_eV', data=E)
        dset.attrs.create('time_stamp', time.time())
        for k in time_to_energy_params:
            dset = h5.create_dataset(k, data=time_to_energy_params[k].value)
            dset.attrs.create('time_stamp', time.time())
        for k in tof_prediction_params:
            dset = h5.require_dataset(k, (), np.float)
            dset[()] = tof_prediction_params[k].value
            dset.attrs.create('time_stamp', time.time())


def open_hdf5_file(file_name, plot=False, verbose=0):
    try:
        # Open the file
        h5 = h5py.File(file_name, 'r+')
    except BaseException as e:
        print 'Could not open the specified hdf5 file "{}".'.format(
            file_name)
        print 'Message was: {}'.format(e.message)
        return -1
    return h5


def get_com(x, y):
    idx_l, idx_h = fwxm(x, y, 0.0, return_data='idx')
    sl = slice(idx_l, idx_h)
    return ((x[sl] * y[sl]).sum()) / (y[sl].sum())


def fwxm(x, y, fraction=0.5, return_data=''):
    y_max = y.max()
    idx_max = y.argmax()

    y_f = y_max * fraction

    for i in range(idx_max, -1, -1):
        if y[i] < y_f:
            idx_low = i
            break
    else:
        idx_low = idx_max

    for i in range(idx_max, len(x)):
        if y[i] < y_f:
            idx_high = i
            break
    else:
        idx_high = idx_max

    if return_data == 'idx':
        return idx_low, idx_high

    if return_data == 'limits':
        return x[idx_low], x[idx_high]

    return (x[idx_low] + x[idx_high]) / 2, x[idx_high] - x[idx_low]


def get_trace_bounds(x, y,
                     threshold=0.0, min_width=2,
                     energy_offset=0,
                     useRel=False, threshold_rel=0.5,
                     roi=slice(None)):

    amp = y[roi]
    scale = x[roi]
    dx = np.mean(np.diff(x))

    if useRel:
        threshold_temp = threshold_rel * np.max(amp[np.isfinite(amp)])
        if threshold_temp < threshold:
            return [np.nan] * 3
        else:
            threshold_V = threshold_temp
    else:
        threshold_V = threshold

    nPoints = np.round(min_width/dx)

    i_min = 0
    for i in range(1, amp.size):
        if amp[i] < threshold_V:
            i_min = i
            continue
        if i-i_min >= nPoints:
            break
    else:
        return [np.nan] * 3

    i_max = amp.size - 1
    for i in range(amp.size-1, -1, -1):
        if amp[i] < threshold_V:
            i_max = i
            continue
        if i_max-i >= nPoints:
            break
    else:
        return [np.nan] * 3

    if i_min == 0 and i_max == amp.size - 1:
        return [np.nan] * 3

    # print 'min =', min, 'max =', max
    val_max = (scale[i_max] + (threshold_V - amp[i_max]) *
               (scale[i_max] - scale[i_max - 1]) /
               (amp[i_max] - amp[i_max - 1]))
    val_min = (scale[i_min] + (threshold_V - amp[i_min]) *
               (scale[i_min + 1] - scale[i_min]) /
               (amp[i_min + 1] - amp[i_min]))

    return val_min, val_max, threshold_V


def update_run_contained_derived_data(file_name, plot=False, verbose=0):
    """Update derived data based on information only in given file.

    Add some derived datasetd to the hdf5 file based on the raw data in the
    file. The added datasets are:

    - Mean of the FEE gas detectors for each shot: fee_mean
    - Maximum TOF waveform signal for each shot: max_signal
    - Frequency spectrum averaged over all shots: fft_spectrum_mean
    - The corresponding frequency axis: fft_freq_axis
    - BC2 energy calculated from the beam position: energy_BC2_MeV
    - L3 energy corrected based on the BC2 energy: energy_L3_corrected_MeV
    """
    if verbose > 0:
        print 'Entering "update_run_contained_derived_data()" ',
        print 'with file_name={}'.format(file_name)

    h5 = open_hdf5_file(file_name, plot, verbose)
    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]

    # Make the fee data set
    raw_fee_dset = raw_group['FEE_energy_mJ']
    fee_mean_dset = make_dataset(h5, 'fee_mean', (n_events,))
    if older(fee_mean_dset, raw_group):
        if verbose > 0:
            print 'Updating fee mean dataset'
        fee_mean_dset[:] = raw_fee_dset[:, 0: 4].mean(1)
        fee_mean_dset.attrs[time_stamp] = time.time()

    # Make max signal dataset
    time_signal_dset = raw_group['time_signal']
    max_sig_dset = make_dataset(h5, 'max_signal', (n_events,))
    if older(max_sig_dset, raw_group):
        if verbose > 0:
            print 'Get the maximum signal for each shot.'
        max_sig_dset[:] = np.max(time_signal_dset, axis=1)
        max_sig_dset.attrs['time_stamp'] = time.time()

    # Make the frequency spectrum
    time_scale = raw_group['time_scale'].value
    spectrum_dset = make_dataset(h5, 'fft_spectrum_mean', time_scale.shape)
    if older(spectrum_dset, [raw_group, max_sig_dset]):
        if verbose > 0:
            print 'Compute the frequency spectrum of the data.'
        max_signal = max_sig_dset.value
        use = max_signal > np.sort(max_signal)[-500:][0]
        signal = time_signal_dset[use, :]
        spectrum_dset[:] = np.mean(np.abs(np.fft.fft(signal, axis=1))**2,
                                   axis=0)
        spectrum_dset.attrs['time_stamp'] = time.time()
    freq_axis_dset = make_dataset(h5, 'fft_freq_axis', time_scale.shape)
    if older(freq_axis_dset, raw_group):
        if verbose > 0:
            print 'Updating the frequency axis.'
        freq_axis_dset[:] = (np.linspace(0., 1e-3, len(time_scale)) /
                             (time_scale[1] - time_scale[0]))
        freq_axis_dset.attrs['time_stamp'] = time.time()

    # Calculate the BC2 energy
    bc2_energy_dset = make_dataset(h5, 'energy_BC2_MeV', (n_events, ))
    if older(bc2_energy_dset, raw_group):
        if verbose > 0:
            print 'Calculating BC2 energy for the bpm reading.'
        # Values comes from a mail from Timothy Maxwell
        # The nominal BC2 energy is 5 GeV (was at least when this data was
        # recorded). The measurement is the relative offset of the beam
        # position in a BPM. The dispersion value is -364.7 mm.
        bc2_energy_dset[:] = 5e3 * (1. - raw_group['position_BC2_mm'][:] /
                                    364.7)
        bc2_energy_dset.attrs['time_stamp'] = time.time()

    # Calculate the corrected L3 energy
    l3_energy_cor_dset = make_dataset(h5, 'energy_L3_corrected_MeV',
                                      (n_events, ))
    if older(l3_energy_cor_dset, [raw_group, bc2_energy_dset,
                                  Timer_object(1434096408)]):
        if verbose > 0:
            print 'Calculating corrected L3 energy.'
        l3_energy_cor_dset[:] = (raw_group['energy_L3_MeV'][:] -
                                 (bc2_energy_dset[:] - 5000))
        l3_energy_cor_dset.attrs['time_stamp'] = time.time()

    # Make the phase cavity time filter
    pct_filter_dset = make_dataset(h5, 'pct_filter', (n_events, ),
                                   dtype=bool)
    if older(pct_filter_dset, [raw_group, Timer_object(0)]):
        pct0 = raw_group['phase_cavity_times'][:, 0]
        pct_filter_dset[:] = (0.4 < pct0) & (pct0 < 1.2)
        pct_filter_dset.attrs[time_stamp] = time.time()

    h5.close()


def update_with_noise_and_response(file_name, plot=False, verbose=0):
    """Update derived data based on noise and response spectra.

    Noise spectrum and detector response are determined form many runs. With
    these spectra a number of new paramters can be derived. These are:

    - snr_spectrum: Signal to Noise ratio spectrum based on the given noise \
    spectrum and the average spectrum in the current run.
    - filtered_time_signal: Wiegner deconvolution of the time signal based on \
    the signal to noise ratio and the detector response function.
    - streak_peak_center: Center of the streaking peak in the sense of the \
    center of mass of the peak in a given ROI. Based on the deconvoluted \
    signal.
    - streak_peak_integral: Photoline intensity by integration of the \
    deconvoluted spectrum in time domain.
    """

    # Make sure that the run contained information is up to date.
    update_run_contained_derived_data(file_name, plot, verbose-1)

    # Open the file.
    h5 = open_hdf5_file(file_name, plot, verbose)
    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]
    time_scale = raw_group['time_scale'].value

    # Make signal to noise ratio.
    snr_dset = make_dataset(h5, 'snr_spectrum', time_scale.shape)
    spectrum_dset = h5['fft_spectrum_mean']
    if older(snr_dset, [spectrum_dset, raw_group, Timer_object(1434015914)]):
        if verbose > 0:
            print 'Updating the signal to noise ratio.',
            print ' In "update_with_noise_and_response()"',
            print ' with file_name={}'.format(file_name)
        snr_dset[:] = construct_snr_spectrum(h5, plot=plot)
        snr_dset.attrs['time_stamp'] = time.time()

    # Deconvolute the response function
    time_signal_dset = raw_group['time_signal']
    deconv_time_signal_dset = make_dataset(h5, 'filtered_time_signal',
                                           time_signal_dset.shape)
    if older(deconv_time_signal_dset, [raw_group, snr_dset]):
        response, t_response = get_response(plot=plot, verbose=verbose-1)
        if verbose > 0:
            print 'Deconvolving traces.'
            print ' In "update_with_noise_and_response()"',
            print ' with file_name={}'.format(file_name),
            print ' {} events to process.'.format(n_events)
        deconvolver = wiener.Deconcolver(snr_dset.value, response)
        for i_evt in range(n_events):
            deconv_time_signal_dset[i_evt, :] = deconvolver.deconvolve(
                time_signal_dset[i_evt, :])
            update_progress(i_evt, n_events, verbose)
        print ''
        deconv_time_signal_dset.attrs['time_stamp'] = time.time()

    # Calculate the center of mass of the streak peak
    time_com_dset = make_dataset(h5, 'streak_peak_center', (n_events, ))
    photo_line_intensity_dset = make_dataset(h5, 'streak_peak_integral',
                                             (n_events, ))
    if older(time_com_dset, [deconv_time_signal_dset,
                             Timer_object(1443006988)]):
        if verbose > 0:
            print 'Calculating streak peak center in time.',
            print ' In "update_with_noise_and_response()"',
            print ' with file_name={}'.format(file_name)

        streak_sl = slice(np.searchsorted(time_scale, streak_time_roi[0]),
                          np.searchsorted(time_scale, streak_time_roi[1],
                                          side='right'))
        time_scale_streak = time_scale[streak_sl]

        ####
        # Center of mass calculation
#        for i_evt in range(n_events):
#            time_com_dset[i_evt] = get_com(
#                time_scale_streak,
#                deconv_time_signal_dset[i_evt, streak_sl])
#            update_progress(i_evt, n_events, verbose)

        ####
        # Fit of Gaussian
        deconv_time_signal = deconv_time_signal_dset.value
        time_com = np.zeros(time_com_dset.shape)
        photo_line_intensity = np.zeros(photo_line_intensity_dset.shape)
        mean_signal = deconv_time_signal[:, streak_sl].mean(axis=0)

        mod = lmfit.models.GaussianModel()
        params = lmfit.Parameters()
        params.add_many(('amplitude', 1, True, 0),
                        ('center', time_scale_streak[np.argmax(mean_signal)],
                         True, min(time_scale_streak), max(time_scale_streak)),
                        ('sigma', 1e-3, True, 0))
        # fit to mean in order to get start parameters for the shot fits
        out = mod.fit(mean_signal, x=time_scale_streak, params=params)
        for k in params:
            params[k].value = out.params[k].value

        for i_evt in range(n_events):
            out = mod.fit(deconv_time_signal[i_evt, streak_sl],
                          params, x=time_scale_streak)
            time_com[i_evt] = out.params['center'].value
            photo_line_intensity[i_evt] = out.params['amplitude'].value
            update_progress(i_evt, n_events, verbose)

        if plot:
            time_scale_streak = time_scale[streak_sl]
            plt.figure('peak finding time domain')
            plt.clf()
            plt.plot(time_scale_streak, mean_signal)
            plt.plot(time_scale_streak, out.best_fit)

        if verbose > 0:
            print ''
        time_com_dset[:] = time_com
        time_com_dset.attrs['time_stamp'] = time.time()
        photo_line_intensity_dset[:] = photo_line_intensity
        photo_line_intensity_dset.attrs['time_stamp'] = time.time()

    h5.close()


def update_with_time_to_energy_conversion(file_name, plot=False, verbose=0):
    """ Make derived data based on time to energy conversion."""

    update_with_noise_and_response(file_name, plot, verbose)

    h5 = open_hdf5_file(file_name, plot, verbose)
    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]

    deconv_time_signal_dset = h5['filtered_time_signal']

    energy_scale_dset = make_dataset(h5, 'energy_scale_eV',
                                     energy_scale_eV.shape)
    energy_trace_dset = make_dataset(h5, 'energy_signal',
                                     (n_events, len(energy_scale_eV)))

    check_tof_to_energy_conversion_matrix(verbose=verbose)
    with h5py.File(tof_to_energy_conversion_file_name, 'r') as tof_to_e_h5:
        if older(energy_scale_dset, [tof_to_e_h5['matrix'],
                                     deconv_time_signal_dset,
                                     Timer_object(1443190000)]):

            if verbose > 0:
                print 'Updating time to energy conversion.',
                print ' In "update_with_time_to_energy_conversion()"',
                print ' with {}'.format(file_name)
            # Get the transformation matrix from file
            M = tof_to_e_h5['matrix'].value
            # Update the energy scale
            energy_scale_dset[:] = tof_to_e_h5['energy_scale_eV'].value
            energy_scale_dset.attrs['time_stamp'] = time.time()
            # Get the photon energy prediction parameters
            params = (tof_to_energy.photon_energy_params() +
                      tof_to_energy.tof_prediction_params())
            for k in params:
                params[k].value = tof_to_e_h5[k].value
            if verbose > 0:
                print 'Computing energy spectra.'
            for i_evt in range(n_events):
                # Energy spectra
                energy_trace_dset[i_evt, :] = M.dot(
                    deconv_time_signal_dset[i_evt, :])
                update_progress(i_evt, n_events, verbose)
            if verbose > 0:
                print ''
            energy_trace_dset.attrs['time_stamp'] = time.time()

    # Calculate energy trace properties
    spectral_properties_group = h5.require_group('spectral_properties')
    spectral_center_dset = make_dataset(spectral_properties_group,
                                        'center_eV', (n_events, ))
    spectral_width_dset = make_dataset(spectral_properties_group,
                                       'width_eV', (n_events, ))
    spectral_threshold_dset = make_dataset(spectral_properties_group,
                                           'threshold', (n_events, ))
    spectral_gaussian_center_dset = make_dataset(spectral_properties_group,
                                                 'gaussian_center',
                                                 (n_events,))

    if older(spectral_center_dset, [energy_trace_dset,
                                    Timer_object(1443421560)]):
        energy_scale = energy_scale_dset[:]
        sl = slice(np.searchsorted(energy_scale, 75),
                   np.searchsorted(energy_scale, 125))
        energy_scale = energy_scale[sl]
        model = lmfit.models.GaussianModel()
        if verbose > 0:
            print 'Calculating spectral center and width:',
            print 'In "update_with_time_to_energy_conversion()"',
            print 'with {}'.format(file_name)
        for i_evt in range(n_events):
            energy_trace = energy_trace_dset[i_evt, sl]
            t_start, t_end, spectral_threshold_dset[i_evt] = \
                get_trace_bounds(energy_scale,
                                 energy_trace,
                                 threshold=8e-5,
                                 min_width=3,
#                                 useRel=True,
#                                 threshold_rel=0.3
                                 )
            center = (t_start + t_end) / 2
            spectral_center_dset[i_evt] = center
            width = t_end - t_start
            spectral_width_dset[i_evt] = width

            # Calculate center of mass
            peak_sl = slice(energy_scale.searchsorted(t_start - width/2),
                            energy_scale.searchsorted(t_end + width/2,
                                                      side='right'))
            peak_trace = energy_trace[peak_sl]
            peak_scale = energy_scale[peak_sl]

#            spectral_com_dset[i_evt] = (np.sum(peak_scale * peak_trace) /
#                                        np.sum(peak_trace))
            if len(peak_trace) > 0:
                out = model.fit(peak_trace, x=peak_scale,
                                center=center, sigma=width/4,
                                amplitude=peak_trace.max() * width / 2)

                spectral_gaussian_center_dset[i_evt] = out.values['center']
            else:
                spectral_gaussian_center_dset[i_evt] = np.nan

            update_progress(i_evt, n_events, verbose)
        spectral_center_dset.attrs['time_stamp'] = time.time()
        spectral_width_dset.attrs['time_stamp'] = time.time()
        spectral_threshold_dset.attrs['time_stamp'] = time.time()
        spectral_gaussian_center_dset.attrs['time_stamp'] = time.time()

    if plot:
        selected_shots = list(np.linspace(0, n_events, 16, endpoint=False))
        plt.figure('peak properties')
        plt.clf()
        _, ax_list = plt.subplots(4, 4, sharex=True, sharey=True,
                                  num='peak properties')
        energy_scale = energy_scale_dset[:]
        sl = slice(np.searchsorted(energy_scale, 75),
                   np.searchsorted(energy_scale, 130))
        energy_scale = energy_scale[sl]
        for i, shot in enumerate(selected_shots):
            energy_trace = energy_trace_dset[shot, :]
            ax = ax_list.flatten()[i]
#            plt.plot(energy_scale - pe_energy_prediction_dset[shot],
            ax.plot(energy_scale, energy_trace[sl])
            c = spectral_center_dset[shot]
            w = spectral_width_dset[shot]
            th = spectral_threshold_dset[shot]
            ax.plot([c-w/2, c+w/2], [th] * 2)

    ##########
    # Calculate electron energy prediction
    e_energy_prediction_params_group = make_group(h5,
                                                  'e_energy_prediction_params')

    if older(e_energy_prediction_params_group, [spectral_gaussian_center_dset,
                                                Timer_object(1443456590)]):
        if verbose > 0:
            print 'Fit the electron energy prediction parameters.',
            print 'In "update_with_time_to_energy_conversion()"',
            print 'with {}'.format(file_name)

        selection = np.isfinite(spectral_gaussian_center_dset.value)
#        &
#                     (0.4 < raw_group['phase_cavity_times'][:, 0]) &
#                     (raw_group['phase_cavity_times'][:, 0] < 1.1))
        spectral_gaussian_center = spectral_gaussian_center_dset[selection]
        if len(spectral_gaussian_center) == 0:
            return

        var_dict = {
            'l3_energy': raw_group['energy_L3_MeV'][selection],
            'bc2_energy': h5['energy_BC2_MeV'][selection],
            # 'fee': h5['fee_mean'][selection],
            'e_energy': spectral_gaussian_center
            }
        prediction_params = \
            tof_to_energy.e_energy_prediction_model_start_params(**var_dict)

        res = lmfit.minimize(tof_to_energy.e_energy_prediction_model,
                             prediction_params,
                             kws=var_dict)

        if verbose > 0:
            print '\nPrediction params:'
            lmfit.report_fit(res)

        # Create or update the parameters from the fit in the group
        for k, v in prediction_params.iteritems():
            d = e_energy_prediction_params_group.require_dataset(
                k, (), np.float)
            d[()] = v.value

        # Remove old parameters that should not be there
        for k in set(e_energy_prediction_params_group.keys()).difference(
                set(prediction_params.keys())):
            del e_energy_prediction_params_group[k]

        e_energy_prediction_params_group.attrs[time_stamp] = time.time()

        if plot:
            deviation = tof_to_energy.e_energy_prediction_model(
                         prediction_params, **var_dict)
            plt.figure('e energy prediction {}'.format(
                h5.filename.split('/')[-1]))
            plt.clf()
            plt.subplot(221)
#            plt.plot(spectral_gaussian_center, deviation, '.')
            plt.scatter(spectral_gaussian_center, deviation,
                        s=4, c=h5['energy_BC2_MeV'][selection],
                        linewidths=(0,), alpha=1)
            plt.xlabel('electron energy (eV)')
            plt.ylabel('prediction residual (eV)')

            x_range = plt.xlim()
            y_range = plt.ylim()
            img, _, _ = np.histogram2d(spectral_gaussian_center, deviation,
                                       bins=2**7, range=[x_range, y_range])
            img = img.T

            plt.subplot(222)
            plt.imshow(img, aspect='auto', interpolation='none',
                       origin='lower', extent=x_range + y_range)

            hist, hist_edges = np.histogram(deviation,
                                            bins=2**5, range=(-3, 3))
            hist_centers = (hist_edges[: -1] + hist_edges[1:])/2
            plt.subplot(223)
            gauss_model = lmfit.models.GaussianModel()
            fit_out = gauss_model.fit(hist, x=hist_centers)
            lmfit.report_fit(fit_out)
            plt.bar(hist_edges[:-1], hist, width=np.diff(hist_edges))
            plt.plot(hist_centers, fit_out.best_fit, 'r', linewidth=2)

            plt.subplot(224)
            plt.plot(spectral_gaussian_center, h5['energy_BC2_MeV'][selection],
                     '.')


def update_with_energy_prediction(file_name, plot=False, verbose=0):

    update_with_time_to_energy_conversion(file_name, plot, verbose)

    h5 = open_hdf5_file(file_name, plot, verbose)
    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]

    prediction_map = {'117': 'h5_files/run118_all.h5',
                      '114': 'h5_files/run115_all.h5',
                      '113': 'h5_files/run112_all.h5',
                      '108': 'h5_files/run109_all.h5'}

    pe_energy_prediction_dset = make_dataset(
        h5, 'photoelectron_energy_prediction_eV', (n_events,))
    spectral_properties_group = h5['spectral_properties']
#    spectral_gaussian_center_dset = spectral_properties_group[
#        'gaussian_center']
    fee_dset = h5['fee_mean']
    energy_BC2_dset = h5['energy_BC2_MeV']
    energy_L3_dset = raw_group['energy_L3_MeV']

    for k, v in prediction_map.iteritems():
        if k in file_name:
            update_with_time_to_energy_conversion(v, plot=False,
                                                  verbose=verbose-1)
            ref_h5 = open_hdf5_file(file_name)
            e_energy_prediction_params_group = \
                ref_h5['e_energy_prediction_params']
            break
    else:
        e_energy_prediction_params_group = h5['e_energy_prediction_params']

    if older(pe_energy_prediction_dset, [e_energy_prediction_params_group,
                                         fee_dset,
                                         energy_BC2_dset,
                                         raw_group,
                                         Timer_object(1443457990)]):

        if verbose > 0:
            print 'Updating energy prediction.',
            print ' In "update_with_energy_prediction()" with {}'.format(
                file_name)
        prediction_params = lmfit.Parameters()
        for k in e_energy_prediction_params_group:
            prediction_params.add(k, e_energy_prediction_params_group[k][()])

        var_dict = {
            'l3_energy': energy_L3_dset.value,
            'bc2_energy': energy_BC2_dset.value,
            'fee': fee_dset.value
            }

        pe_energy_prediction_dset[:] = tof_to_energy.e_energy_prediction_model(
            prediction_params, **var_dict)

        pe_energy_prediction_dset.attrs[time_stamp] = time.time()

    ##########
    # Make the christmas three histogram
    n_spectral_center_bins = 2**7
    n_spectral_width_bins = 2**7
    spectral_center_axis_dset = make_dataset(spectral_properties_group,
                                             'center_axis_eV',
                                             (n_spectral_center_bins, ))
    spectral_width_axis_dset = make_dataset(spectral_properties_group,
                                            'width_axis_eV',
                                            (n_spectral_width_bins, ))
    spectral_histogram_dset = make_dataset(spectral_properties_group,
                                           'histogram',
                                           (n_spectral_width_bins,
                                            n_spectral_center_bins))

    spectral_center_dset = spectral_properties_group['center_eV']
    spectral_width_dset = spectral_properties_group['width_eV']
    pct_filter_dset = h5['pct_filter']

    if older(spectral_histogram_dset, [spectral_center_dset,
                                       spectral_width_dset,
                                       pe_energy_prediction_dset,
                                       pct_filter_dset,
                                       Timer_object(1443603690)]):
        if verbose > 0:
            print 'Making the christmas tree plot.',
            print ' In "update_with_energy_prediction()"',
            print ' with {}'.format(file_name)
        spectral_width_axis_dset[:] = np.linspace(0, 35, n_spectral_width_bins)
        spectral_width_axis_dset.attrs['time_stamp'] = time.time()
        spectral_center_axis_dset[:] = np.linspace(-12, 12,
                                                   n_spectral_center_bins)
        spectral_center_axis_dset.attrs['time_stamp'] = time.time()

        I = pct_filter_dset.value & (-0.1 <
                                     raw_group['phase_cavity_times'][:, 1])

        hist = aol_plotting.center_histogram_2d(
            spectral_center_dset[I] - pe_energy_prediction_dset[I],
            spectral_width_dset[I],
            spectral_center_axis_dset[:],
            spectral_width_axis_dset[:])
        hist[hist == 0] = np.nan
        spectral_histogram_dset[:] = hist
        spectral_histogram_dset.attrs['time_stamp'] = time.time()

    if plot:
        plt.figure('christmas tree {}'.format(h5.filename.split('/')[-1]))
        plt.clf()
        plt.imshow(spectral_histogram_dset[:], aspect='auto',
                   interpolation='none', origin='lower',
                   extent=(np.min(spectral_center_axis_dset),
                           np.max(spectral_center_axis_dset),
                           np.min(spectral_width_axis_dset),
                           np.max(spectral_width_axis_dset)))

        plt.xlabel('center (eV)')
        plt.ylabel('width (eV)')
        plt.colorbar()
        plt.savefig('figures/christmas_tree_{}.png'.format(
            h5.filename.split('/')[-1].split('.')[0]))

    h5.close()


def load_file(file_name, plot=False, verbose=0):
    """ Load file and make sure it is up to date."""
#    if verbose > 0:
#        print 'Entering "load_file()" with file_name={}'.format(file_name)

    update_with_energy_prediction(file_name, plot, verbose)

    h5 = open_hdf5_file(file_name, plot, verbose)
    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]

    if verbose > 0:
        print 'File {} processed.'.format(h5.file)
        print 'It contains', n_events, 'events.'
    if verbose > 1:
        list_hdf5_content(h5)

    return h5


def touch_all_files(verbose=2):
    file_names = ['/'.join([data_dir, f]) for f in os.listdir(data_dir) if
                  f.startswith('run') and f.endswith('_all.h5')]

    for name in file_names:
        load_file(name, verbose=verbose)


if __name__ == '__main__':
    # Parset the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--hdf5_file', type=str,
                        default='h5_files/run108_all.h5',
                        help='Path to hdf5 file to process')
    parser.add_argument('--plot', action='store_true',
                        help='Display plots. Default: no plots.')
    parser.add_argument('-v', '--verbose', action='count',
                        help='increase output verbosity')
    args = parser.parse_args()

    # Unpack the parser arguments.
    hdf5_file = args.hdf5_file
    plot = args.plot
    verbose = args.verbose

    # If plotting is requested, ryn pyplot in the interactive mode.
    if plot:
        plt.ion()

    if verbose > 0:
        print 'Get the noise spectrum just to make sure it is up to date.'
    get_nois_spectrum(plot=plot, verbose=verbose)

    # Load the given file.
    if verbose > 0:
        print 'Load the requested file: {}'.format(hdf5_file)
    h5 = load_file(hdf5_file, verbose=verbose, plot=plot)

    # Get the raw group of the file.
    raw_group = h5['raw']

    # Number of events in the file.
    n_events = len(raw_group['event_time_s'])

    # Time trace rellated information.
    raw_time = raw_group['time_scale'].value
    raw_traces_dset = raw_group['time_signal']
    filtered_traces = h5['filtered_time_signal']

    # Pulse energy
    raw_fee_dset = raw_group['FEE_energy_mJ']
    n_fee = raw_fee_dset.shape[1]

    # frequency domain
    freq_axis = h5['fft_freq_axis'].value
    fft_mean = h5['fft_spectrum_mean'].value
    snr = h5['snr_spectrum'].value

    if plot and False:
        if verbose > 0:
            print 'Plotting fee correlations.'
        plt.figure('fee')
        plt.clf()
        ax = None
        for i in range(n_fee):
            for k in range(n_fee):
                ax = plt.subplot(n_fee, n_fee, i + k*n_fee + 1,
                                 sharex=ax, sharey=ax)
                ax.plot(raw_fee_dset[:, i], raw_fee_dset[:, k], '.')
                if i > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if k < n_fee-1:
                    plt.setp(ax.get_xticklabels(), visible=False)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)

        if verbose > 0:
            print 'Plotting fee histogram.'
        plt.figure('fee histogram')
        plt.clf()
        plt.hist(h5['fee_mean'].value, bins=100)

    if plot:
        if verbose > 0:
            print 'Plot signal maximium histogram.'
        plt.figure('signal hist')
        plt.clf()
        plt.hist(h5['max_signal'], bins=100)

    if plot:
        if verbose > 0:
            print 'Plot spectr'
        plt.figure('fft')
        plt.clf()
        plt.semilogy(freq_axis, fft_mean, label='average spectrum')
        plt.semilogy(freq_axis, snr, label='snr')
        plt.legend(loc='best')

    # Plot some traces
    if plot:
        if verbose > 0:
            print 'Plotting traces'
        trace_fig = plt.figure('traces {}'.format(hdf5_file))
        trace_fig.clf()
        raw_mean_tr = raw_traces_dset.value.mean(0)
        deconv_mean_tr = filtered_traces.value.mean(0)
        rand_event = np.random.randint(n_events)
        response, _ = get_response(plot=False, verbose=verbose)

        plt.plot(raw_time, raw_traces_dset[rand_event, :],
                 label='single trace')
        plt.plot(raw_time, filtered_traces[rand_event, :],
                 label='Deconv single trace')

        plt.plot(raw_time, raw_mean_tr, label='mean trace')
        plt.plot(raw_time, deconv_mean_tr,
                 label='Deconv mean')
        plt.legend(loc='best')

    # Plot the phase cavity times
    pct = raw_group['phase_cavity_times']
    plt.figure('Phase cavity times')
    plt.clf()
#    pc_selection = (np.isfinite(np.sum(pct, axis=1)) &
#                    (pct[:, 0] > -2) & (pct[:, 0] < 2) &
#                    (pct[:, 1] > -2) & (pct[:, 1] < 2))
#                    (pct[:, 0] > -50) & (pct[:, 0] < 50))
    pc_selection = h5['pct_filter'].value

    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.title('Time {}'.format(i))
        hist, hist_edges = np.histogram(pct[pc_selection, i], bins=100)
        plt.bar(hist_edges[: -1], hist, width=np.diff(hist_edges))

    plt.subplot(133)
    plt.plot(pct[pc_selection, 0], pct[pc_selection, 1], '.')

    # Plot energy traces and photon energy diagnostics
    pe_energy_dset = h5['photoelectron_energy_prediction_eV']
    energy_scale = h5['energy_scale_eV'][:]
    energy_signal_dset = h5['energy_signal']
    selected_shots = np.linspace(0, n_events, 100, endpoint=False, dtype=int)
    plt.figure('Energy spectra')
    plt.clf()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    dy = 1e-5
    for i, shot in enumerate(selected_shots):
        ax1.plot(energy_scale, energy_signal_dset[shot, :] + dy * i)
        ax2.plot(energy_scale - pe_energy_dset[shot],
                 energy_signal_dset[shot, :] + dy * i)
    ax2.set_xlim(-20, 25)
