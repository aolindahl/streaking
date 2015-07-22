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
nois_file_name = data_dir + '/nois.h5'
tof_to_energy_conversion_file_name = data_dir + '/time_to_energy.h5'


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
        if verbose > 0:
            print 'Get the response data.'
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
        update_internal_parameters(file_name, verbose=verbose)
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
    if verbose:
        print 'Entering get_nois_spectrum()'
    try:
        if verbose:
            print 'Get the signal to nois ratio.'
        with h5py.File(nois_file_name, 'r') as f:
            pass
        new_noise = False
    except IOError:
        if verbose > 0:
            print 'Could not open response file. Trying to make it.'
        construct_nois_spectrum(plot=plot, verbose=verbose)
        new_noise = True

    if not new_noise:
        make_new_noise = False
        with h5py.File(nois_file_name, 'r') as f:
            nois = f['nois']
            h5_file_names = get_file_names_for_noise_spectrum()
            for h5_name in h5_file_names:
                with h5py.File(h5_name, 'r') as h5:
                    if older(nois, h5['raw']):
                        make_new_noise = True
                        if verbose:
                            print 'Nois was made earlier than the raw data in',
                            print 'the file', h5_name, 'Make new nois.'
                        break
                    elif False:
                        print 'Nois was made later than the raw data in',
                        print 'the file', h5_name

        if make_new_noise:
            construct_nois_spectrum(plot=plot, verbose=verbose)

    with h5py.File(nois_file_name, 'r') as f:
        nois = f['nois']
        return nois.value, nois.attrs['time_stamp']


def construct_nois_spectrum(plot=False, verbose=0):
    h5_file_names = get_file_names_for_noise_spectrum()
    for file_name in h5_file_names:
        update_internal_parameters(file_name)
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
            if verbose:
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
        dset = f.create_dataset('nois', data=wt_amp)
        dset.attrs.create('time_stamp', time.time())
        f.create_dataset('freq', data=freq)

    return get_nois_spectrum()


def construct_snr_spectrum(h5, plot=False):
    nois, t = get_nois_spectrum()

    sig_spec = h5['fft_spectrum_mean'].value
    freq = h5['fft_freq_axis'].value

    wt_spec = wavelet_filter.wavelet_filt(sig_spec, thresh=wt_th)
    wt_spec[1:] = (wt_spec[1:] + wt_spec[-1:0:-1]) / 2

    snr = (wt_spec - nois) / nois

    if plot:
        plt.figure('signal and nois')
        plt.clf()
        plt.semilogy(freq, sig_spec, label='signal')
        plt.semilogy(freq, nois, label='nois')
        plt.semilogy(freq, wt_spec, label='wt signal')
        plt.semilogy(freq, snr, label='snr')
        plt.legend(loc='best')

    return snr


def check_tof_to_energy_conversion_matrix(plot=False, verbose=0):
    try:
        if verbose:
            print 'Load time to energy converstion.'
        with h5py.File(tof_to_energy_conversion_file_name, 'r'):
            pass
    except IOError:
        if verbose:
            print 'Could not open the file. Making the conversion matrix.'
        construc_tof_to_energy_conversion_matrix(plot=plot, verbose=verbose)

    _, h5_dict, _ = tof_to_energy.load_tof_to_energy_data(verbose=verbose)
    with h5py.File(tof_to_energy_conversion_file_name, 'r') as trans_h5:
        if not older(
                trans_h5['matrix'],
                [h5['streak_peak_integral'] for h5 in h5_dict.itervalues()] +
                [Timer_object(1437117486)]):
            return
    if verbose:
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
    dx = np.mean(np.diff(x))

    if useRel:
        threshold_temp = threshold_rel * np.max(amp[np.isfinite(amp)])
        if threshold_temp < threshold:
            return [np.nan] * 3
        else:
            threshold_V = threshold_temp
    nPoints = np.round(min_width/dx)

    min = 0
    for i in range(1, amp.size):
        if amp[i] < threshold_V:
            min = i
            continue
        if i-min >= nPoints:
            break
    else:
        return [np.nan] * 3

    max = amp.size - 1
    for i in range(amp.size-1, -1, -1):
        if amp[i] < threshold_V:
            max = i
            continue
        if max-i >= nPoints:
            break
    else:
        return [np.nan] * 3

    if min == 0 and max == amp.size - 1:
        return [np.nan] * 3

    # print 'min =', min, 'max =', max
    min = x[roi][min]
    max = x[roi][max]

    return min, max, threshold_V


def update_internal_parameters(file_name, plot=False, verbose=0):
    if verbose:
        print 'Entering "update_internal_parameters()" ',
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
        if verbose:
            print 'Compute the frequency spectrum of the data.'
        max_signal = max_sig_dset.value
        use = max_signal > np.sort(max_signal)[-500:][0]
        signal = time_signal_dset[use, :]
        spectrum_dset[:] = np.mean(np.abs(np.fft.fft(signal, axis=1))**2,
                                   axis=0)
        spectrum_dset.attrs['time_stamp'] = time.time()
    freq_axis_dset = make_dataset(h5, 'fft_freq_axis', time_scale.shape)
    if older(freq_axis_dset, raw_group):
        if verbose:
            print 'Updating the frequency axis.'
        freq_axis_dset[:] = (np.linspace(0., 1e-3, len(time_scale)) /
                             (time_scale[1] - time_scale[0]))
        freq_axis_dset.attrs['time_stamp'] = time.time()

    # Calculate the BC2 energy
    bc2_energy_dset = make_dataset(h5, 'energy_BC2_MeV', (n_events, ))
    if older(bc2_energy_dset, raw_group):
        if verbose:
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
        if verbose:
            print 'Calculating corrected L3 energy.'
        l3_energy_cor_dset[:] = (raw_group['energy_L3_MeV'][:] -
                                 (bc2_energy_dset[:] - 5000))
        l3_energy_cor_dset.attrs['time_stamp'] = time.time()

    h5.close()


def update_with_noise_and_response(file_name, plot=False, verbose=0):
    if verbose:
        print 'Entering "update_with_noise_and_response()" ',
        print 'with file_name={}'.format(file_name)

    update_internal_parameters(file_name, plot, verbose)

    h5 = open_hdf5_file(file_name, plot, verbose)
    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]
    time_scale = raw_group['time_scale'].value

    # Make signal to nois ratio
    snr_dset = make_dataset(h5, 'snr_spectrum', time_scale.shape)
    spectrum_dset = h5['fft_spectrum_mean']
    if older(snr_dset, [spectrum_dset, raw_group, Timer_object(1434015914)]):
        if verbose:
            print 'Updating the signal to nois ratio.'
        snr_dset[:] = construct_snr_spectrum(h5, plot=plot)
        snr_dset.attrs['time_stamp'] = time.time()

    # Deconvolute the response function
    time_signal_dset = raw_group['time_signal']
    deconv_time_signal_dset = make_dataset(h5, 'filtered_time_signal',
                                           time_signal_dset.shape)
    if older(deconv_time_signal_dset, [raw_group, snr_dset]):
        response, t_response = get_response(plot=plot, verbose=verbose-1)
        if verbose:
            print 'Deconvolving traces for {}.'.format(h5.filename),
            print '{} events to process.'.format(n_events)
        deconvolver = wiener.Deconcolver(snr_dset.value, response)
        for i_evt in range(n_events):
            deconv_time_signal_dset[i_evt, :] = deconvolver.deconvolve(
                time_signal_dset[i_evt, :])
            update_progress(i_evt, n_events, verbose)
        print ''
        deconv_time_signal_dset.attrs['time_stamp'] = time.time()

    # Calculate the center of mass of the streak peak
    time_com_dset = make_dataset(h5, 'streak_peak_center', (n_events, ))
    if older(time_com_dset, [deconv_time_signal_dset,
                             Timer_object(1437488661)]):
        if verbose:
            print 'Calculating streak peak center in time.'
        streak_sl = slice(np.searchsorted(time_scale, streak_time_roi[0]),
                          np.searchsorted(time_scale, streak_time_roi[1],
                                          side='right'))
        time_scale_streak = time_scale[streak_sl]
        for i_evt in range(n_events):
            time_com_dset[i_evt] = get_com(
                time_scale_streak,
                deconv_time_signal_dset[i_evt, streak_sl])
            update_progress(i_evt, n_events, verbose)
        if verbose:
            print ''
        time_com_dset.attrs['time_stamp'] = time.time()

    # Calculate the photo line intensity
    photo_line_intensity_dset = make_dataset(h5, 'streak_peak_integral',
                                             (n_events, ))
    if older(photo_line_intensity_dset, deconv_time_signal_dset):
        if verbose:
            print 'Calculating photo line integral in time domain.'
        for i_evt in range(n_events):
            streak_sig = deconv_time_signal_dset[i_evt, streak_sl]
            center, width = fwxm(time_scale_streak,
                                 streak_sig,
                                 fraction=0.5)
            lo = center - width
            hi = center + width
            t_start = np.searchsorted(time_scale_streak, lo)
            t_end = np.searchsorted(time_scale_streak, hi)
            t = np.empty((2 + t_end - t_start, ))
            t[[0, -1]] = lo, hi
            t[1:-1] = time_scale_streak[t_start: t_end]
            sig = np.empty_like(t)
            sig[[0, -1]] = np.interp(t[[0, -1]], time_scale_streak,
                                     streak_sig)
            sig[1:-1] = streak_sig[t_start:t_end]
            photo_line_intensity_dset[i_evt] = np.trapz(sig, t)
            update_progress(i_evt, n_events, verbose)
        if verbose:
            print ''
        photo_line_intensity_dset.attrs['time_stamp'] = time.time()

    h5.close()


def load_file(file_name, plot=False, verbose=0):
    if verbose:
        print 'Entering "load_file()" with file_name={}'.format(file_name)

    update_with_noise_and_response(file_name, plot, verbose)

    h5 = open_hdf5_file(file_name, plot, verbose)
    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]

    deconv_time_signal_dset = h5['filtered_time_signal']

    energy_scale_dset = make_dataset(h5, 'energy_scale_eV',
                                     energy_scale_eV.shape)
    energy_trace_dset = make_dataset(h5, 'energy_signal',
                                     (n_events, len(energy_scale_eV)))
    pe_energy_prediction_dset = make_dataset(
        h5, 'photoelectron_energy_prediction_eV', (n_events,))

    if verbose:
        print 'Checking the time to energy conversion time stamp validity.'
    check_tof_to_energy_conversion_matrix(verbose=verbose-1)
    with h5py.File(tof_to_energy_conversion_file_name, 'r') as tof_to_e_h5:
        if (older(energy_scale_dset, [tof_to_e_h5['matrix'],
                                      deconv_time_signal_dset,
                                      Timer_object(1437058277)]) or
            older(pe_energy_prediction_dset, [tof_to_e_h5['matrix'],
                                              deconv_time_signal_dset,
                                              Timer_object(1437058277)])):
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
            if verbose:
                print 'Computing energy spectra.'
            for i_evt in range(n_events):
                # Energy spectra
                energy_trace_dset[i_evt, :] = M.dot(
                    deconv_time_signal_dset[i_evt, :])
                update_progress(i_evt, n_events, verbose)
            if verbose:
                print ''
            energy_trace_dset.attrs['time_stamp'] = time.time()

            if verbose:
                print 'Computing photon energy.'
            pe_energy_prediction_dset[:] = \
                tof_to_energy.photoelectron_energy_prediction_model(
                    params,
                    l3_energy=raw_group['energy_L3_MeV'].value,
                    bc2_energy=h5['energy_BC2_MeV'].value,
                    fee=h5['fee_mean'].value)
            pe_energy_prediction_dset.attrs['time_stamp'] = time.time()

    # Calculate energy trace properties
    spectral_properties_group = h5.require_group('spectral_properties')
    spectral_center_dset = make_dataset(spectral_properties_group,
                                        'center_eV', (n_events, ))
    spectral_width_dset = make_dataset(spectral_properties_group,
                                       'width_eV', (n_events, ))
    spectral_threshold_dset = make_dataset(spectral_properties_group,
                                           'threshold', (n_events, ))

    if older(spectral_center_dset, [energy_trace_dset,
                                    Timer_object(1437379331)]):
        energy_scale = energy_scale_dset[:]
        sl = slice(np.searchsorted(energy_scale, 75),
                   np.searchsorted(energy_scale, 125))
        energy_scale = energy_scale[sl]
        if verbose:
            'Calculating spectral center and width:'
        for i_evt in range(n_events):
            photon_energy = pe_energy_prediction_dset[i_evt]
            t_start, t_end, spectral_threshold_dset[i_evt] = \
                get_trace_bounds(energy_scale - photon_energy,
                                 energy_trace_dset[i_evt, sl],
                                 threshold=1e-4,
                                 min_width=2,
                                 useRel=True,
                                 threshold_rel=0.4)
            spectral_center_dset[i_evt] = (t_start + t_end) / 2
            spectral_width_dset[i_evt] = t_end - t_start
            update_progress(i_evt, n_events, verbose)
        spectral_center_dset.attrs['time_stamp'] = time.time()
        spectral_width_dset.attrs['time_stamp'] = time.time()
        spectral_threshold_dset.attrs['time_stamp'] = time.time()

    if plot:
        selected_shots = list(np.linspace(0, n_events, 16, endpoint=False))
        plt.figure('peak properties')
        plt.clf()
        energy_scale = energy_scale_dset[:]
        sl = slice(np.searchsorted(energy_scale, 75),
                   np.searchsorted(energy_scale, 125))
        energy_scale = energy_scale[sl]
        for i, shot in enumerate(selected_shots):
            energy_trace = energy_trace_dset[shot, :]
            plt.subplot(4, 4, i+1)
            plt.plot(energy_scale - pe_energy_prediction_dset[shot],
                     energy_trace[sl])
            c = spectral_center_dset[shot]
            w = spectral_width_dset[shot]
            th = spectral_threshold_dset[shot]
            plt.plot([c-w/2, c+w/2], [th] * 2)

    n_spectral_center_bins = 2**6
    n_spectral_width_bins = 2**6
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

    if older(spectral_histogram_dset, [spectral_center_dset,
                                       spectral_width_dset,
                                       Timer_object(1437380252)]):
        if verbose:
            print 'Making the christmas tree plot.'
        # Make the spectral properties histogram
#        spectral_width_axis_dset[:] = np.linspace(
#            np.nanmin(spectral_width_dset),
#            np.nanmax(spectral_width_dset),
#            n_spectral_width_bins)
        spectral_width_axis_dset[:] = np.linspace(0, 25, n_spectral_width_bins)
        spectral_width_axis_dset.attrs['time_stamp'] = time.time()
#        spectral_center_axis_dset[:] = np.linspace(
#            np.nanmin(spectral_center_dset),
#            np.nanmax(spectral_center_dset),
#            n_spectral_center_bins)
        spectral_center_axis_dset[:] = np.linspace(-10, 10,
                                                   n_spectral_center_bins)
        spectral_center_axis_dset.attrs['time_stamp'] = time.time()

        hist = aol_plotting.center_histogram_2d(
            spectral_center_dset[:], spectral_width_dset[:],
            spectral_center_axis_dset[:], spectral_width_axis_dset[:])
        hist[hist == 0] = np.nan
        spectral_histogram_dset[:] = hist
        spectral_histogram_dset.attrs['time_stamp'] = time.time()

    if plot:
        plt.figure('christmas tree {}'.format(h5.filename.split('/')[-1]))
        plt.clf()
        ax = plt.imshow(spectral_histogram_dset[:], aspect='auto',
                        interpolation='none', origin='lower',
                        extent=(np.min(spectral_center_axis_dset),
                                np.max(spectral_center_axis_dset),
                                np.min(spectral_width_axis_dset),
                                np.max(spectral_width_axis_dset)))
        plt.colorbar()

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
    # Parset the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--hdf5_file', type=str,
                        default='h5_files/run108_all.h5',
                        help='Path to hdf5 file to process')
    parser.add_argument('--plot', action='store_true',
                        help='Display plots. Default: no plots.')
    parser.add_argument('-v', '--verbose', action='count',
                        help='increase output verbosity')
    args = parser.parse_args()

    hdf5_file = args.hdf5_file
    plot = args.plot
    verbose = args.verbose

    if plot:
        plt.ion()

    if verbose:
        print 'Get the nois spectrum just to make sure it is up to date.'
    get_nois_spectrum(plot=plot, verbose=verbose)

    if verbose:
        print 'Load the requested file: {}'.format(hdf5_file)
    h5 = load_file(hdf5_file, verbose=verbose, plot=plot)

    # Get the raw group
    raw_group = h5['raw']

    # Number of events in the file
    n_events = len(raw_group['event_time_s'])

    # Time trace rellated information
    raw_time = raw_group['time_scale'].value
    raw_traces = raw_group['time_signal']
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
        if verbose:
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
        raw_mean_tr = raw_traces.value.mean(0)
        deconv_mean_tr = filtered_traces.value.mean(0)
        rand_event = np.random.randint(n_events)
        response, _ = get_response(plot=False, verbose=verbose)

        plt.plot(raw_time, raw_traces[rand_event, :],
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
    pc_selection = (np.isfinite(np.sum(pct, axis=1)) &
                    (pct[:, 0] > -2) & (pct[:, 0] < 2) &
                    (pct[:, 1] > -2) & (pct[:, 1] < 2))
#                    (pct[:, 0] > -50) & (pct[:, 0] < 50))

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
