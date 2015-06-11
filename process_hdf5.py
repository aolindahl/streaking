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

from aolPyModules import wiener, wavelet_filter

prompt_roi = [1.508, 1.535]
wt_th = 0.03
time_stamp = 'time_stamp'
data_dir = 'h5_files'
response_file_name = data_dir + '/response.h5'
nois_file_name = data_dir + '/nois.h5'


def update_progress(progress):
    num_squares = 40
    base_string = '\r[{:' + str(num_squares) + '}] {}%'
    print base_string.format('#'*(progress * num_squares / 100), progress),


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

    return np.any([dset.attrs[time_stamp] < d.attrs[time_stamp]
                   for d in dset_list])


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
    file_name = 'h5_files/run{}_all.h5'
    if verbose > 0:
        print 'Loading Kr files for prompt determination.'
    h5_list = [load_file(file_name.format(run), verbose=verbose-1) for
               run in runs]
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
                    elif verbose:
                        print 'Nois was made later than the raw data in',
                        print 'the file', h5_name

        if make_new_noise:
            construct_nois_spectrum(plot=plot, verbose=verbose)

    with h5py.File(nois_file_name, 'r') as f:
        nois = f['nois']
        return nois.value, nois.attrs['time_stamp']


def construct_nois_spectrum(plot=False, verbose=0):
    h5_file_names = get_file_names_for_noise_spectrum()
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

    snr = wt_spec / nois

    if plot:
        plt.figure('signal and nois')
        plt.clf()
        plt.semilogy(freq, sig_spec)
        plt.semilogy(freq, nois)
        plt.semilogy(freq, wt_spec)
        plt.semilogy(freq, snr)

    return snr


def load_file(file_name, verbose=0):
    if verbose:
        print 'Entering "load_file()" with file_name={}'.format(file_name)
    try:
        # Open the file
        h5 = h5py.File(file_name, 'r+')
    except BaseException as e:
        print 'Could not open the specified hdf5 file "{}".'.format(
            file_name)
        print 'Message was: {}'.format(e.message)
        return -1

    raw_group = h5['raw']
    n_events = raw_group['event_time_s'].shape[0]
    raw_fee_dset = raw_group['FEE_energy_mJ']
    time_scale = raw_group['time_scale'].value
    time_signal_dset = raw_group['time_signal']

    if verbose > 0:
        print 'File {} opened.'.format(h5.file)
        print 'I contains', n_events, 'events.'
    if verbose > 1:
        list_hdf5_content(h5)

    # Make the fee data set
    fee_mean_dset = make_dataset(h5, 'fee_mean', (n_events,))
    if older(fee_mean_dset, raw_group):
        if verbose > 0:
            print 'Updating fee mean dataset'
        fee_mean_dset[:] = raw_fee_dset[:, 0: 4].mean(1)
        fee_mean_dset.attrs[time_stamp] = time.time()

    max_sig_dset = make_dataset(h5, 'max_signal', (n_events,))
    if older(max_sig_dset, raw_group):
        if verbose > 0:
            print 'Get the maximum signal for each shot.'
        max_sig_dset[:] = np.max(time_signal_dset, axis=1)
        max_sig_dset.attrs['time_stamp'] = time.time()

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
        freq_axis_dset[:] = (np.linspace(0., 1e-3, len(time_scale)) /
                             (time_scale[1] - time_scale[0]))
        freq_axis_dset.attrs['time_stamp'] = time.time()

    snr_dset = make_dataset(h5, 'snr_spectrum', time_scale.shape)
    if older(snr_dset, [spectrum_dset, raw_group]):
        snr_dset[:] = construct_snr_spectrum(h5)
        snr_dset.attrs['time_stamp'] = time.time()

    deconv_time_signal_dset = make_dataset(h5, 'filtered_time_signal',
                                           time_signal_dset.shape)
    if older(deconv_time_signal_dset, [raw_group, snr_dset]):
        if verbose:
            print 'Deconvolving traces for {}.'.format(h5.filename)
            print '{} events to process.'.format(n_events)
        response, t_response = get_response(plot=plot, verbose=verbose)
        deconvolver = wiener.Deconcolver(snr_dset.value, response)
        for i_evt in range(n_events):
            deconv_time_signal_dset[i_evt, :] = deconvolver.deconvolve(
                time_signal_dset[i_evt, :])

            if (verbose and
                    ((i_evt % (n_events/20) == 0) or (i_evt == n_events-1))):
                update_progress(100 * i_evt / (n_events-1))
        print ''
        deconv_time_signal_dset.attrs['time_stamp'] = time.time()

    return h5


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
    h5 = load_file(hdf5_file, verbose=verbose)

    # Get the raw group
    raw_group = h5['raw']

    # Number of events in the file
    n_events = len(raw_group['event_time_s'])

    # Time trace rellated information
    raw_time = raw_group['time_scale'].value
    raw_traces = raw_group['time_signal']

    # Pulse energy
    raw_fee_dset = raw_group['FEE_energy_mJ']
    n_fee = raw_fee_dset.shape[1]

    # Get the prompt signal for deconvolution
    response, t_response = get_response(plot=plot, verbose=verbose)

    snr = h5['snr_spectrum']

    if plot:
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

    # Plot some traces
    if plot:
        if verbose > 0:
            print 'Plotting traces'
        trace_fig = plt.figure('traces {}'.format(hdf5_file))
        trace_fig.clf()
        raw_mean_tr = raw_traces.value.mean(0)
        rand_event = np.random.randint(n_events)
        plt.plot(raw_time, raw_traces[rand_event, :],
                 label='single trace')
        plt.plot(raw_time, raw_mean_tr, label='mean trace')
        plt.plot(raw_time, wiener.deconvolution(raw_mean_tr,
                                                snr,
                                                response),
                 label='Deconv mean')
        plt.plot(raw_time, wiener.deconvolution(raw_traces[rand_event, :],
                                                snr,
                                                response),
                 label='Deconv single')
        plt.legend(loc='best')
