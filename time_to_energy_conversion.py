# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:49:41 2015

@author: antlin
"""
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import h5py

import process_hdf5 as process
from aolPyModules import tof as tof_module
from aolPyModules.aolUtil import struct

Ne1s = 870.2


def tof_prediction_model(params, l3_energy, bc2_energy=None, fee=None,
                         tof=None):
    d_eff = params['d_eff_prediction'].value
    t_0 = params['t_0_prediction'].value
    E_0 = params['E_0_prediction'].value
    K = params['K'].value
    IP = params['IP'].value
    BC2_nominal = params['BC2_nominal'].value
    BC2_factor = params['BC2_factor'].value
    fee_factor = params['fee_factor'].value

    if bc2_energy is None:
        e_beam_energy = l3_energy
    else:
        e_beam_energy = l3_energy - BC2_factor * (bc2_energy - BC2_nominal)

    if fee is not None:
        e_beam_energy -= fee * fee_factor

    mod = t_0 + d_eff / np.sqrt(K * e_beam_energy**2 - IP - E_0)

    if tof is None:
        return mod
    return mod-tof


def tof_prediction_params(l3_energy=[], bc2_energy=[], fee=[],
                          tof=[]):
    params = photon_energy_params(postfix='_prediction')
    params.add('K', 4.57e-5, min=4.5e-5, max=1e-4)

    use_bc2 = len(bc2_energy) > 0
    params.add('BC2_nominal', 5e3, min=4.9e3, max=5.1e3, vary=use_bc2)
    params.add('BC2_factor', 1, min=0, max=2, vary=use_bc2)

    use_fee = len(fee) > 0
    params.add('fee_factor', 108 if use_fee else 0, vary=use_fee)

    return params


def photoelectron_energy_model(params, tof, e_pe=None, postfix=''):
    t_0 = params['t_0' + postfix].value
    E_0 = params['E_0' + postfix].value
    d_eff = params['d_eff' + postfix].value

    mod = E_0 + (d_eff / (tof - t_0))**2

    if e_pe is None:
        return mod
    return mod - e_pe


def photon_energy_model(params, tof, e_p=None, postfix=''):
    e_ip = params['IP'].value

    mod = photoelectron_energy_model(params, tof, postfix=postfix) + e_ip

    if e_p is None:
        return mod
    return mod - e_p


def photon_energy_params(postfix=''):
    params = lmfit.Parameters()
    params.add('d_eff' + postfix, 1.23, min=1, max=2)
    params.add('t_0' + postfix, 1.49, min=1.47, max=1.51)
    params.add('E_0' + postfix, 0, vary=False)
    params.add('IP', Ne1s, vary=False)

    return params


def photoelectron_energy_prediction_model(params, l3_energy, bc2_energy=None,
                                          fee=None, tof=None):
    return photoelectron_energy_model(params,
                                      tof_prediction_model(
                                          params,
                                          l3_energy=l3_energy,
                                          bc2_energy=bc2_energy,
                                          fee=fee))


def fel_params():
    params = lmfit.Parameters()
    params.add('K', 1)
    return params


def fel_model(params, e_energy, p_energy=None):
    K = params['K'].value
    mod = K * e_energy**2
    if p_energy is None:
        return mod
    return mod - p_energy


def fel_model_inverted(params, p_energy, e_energy=None):
    K = params['K'].value
    mod = np.sqrt(p_energy / K)
    if e_energy is None:
        return mod
    return mod - e_energy


def tof_e_energy_params(d_exp_list=[2]):
    params = lmfit.Parameters()
    params.add('E_0', 0)
    params.add('t_0', 1.5, min=1.0, max=2.0)
    for d_exp in d_exp_list:
        params.add('d_{}'.format(d_exp), 1 if d_exp == 2 else 0)
    return params


def tof_e_energy_model(params, tof, energy=None):
    E_0 = params['E_0'].value
    t_0 = params['t_0'].value
    d = {}
    for d_exp in range(1, 11):
        name = 'd_{}'.format(d_exp)
        if name in params.keys():
            d[d_exp] = params[name].value

    mod = E_0 * np.ones_like(tof)
    for d_exp, d_val in d.iteritems():
        mod += d_val / (tof - t_0)**d_exp

    if energy is None:
        return mod
    return mod - energy


def tof_photon_energy_params(d_exp_list=[2]):
    params = tof_e_energy_params(d_exp_list=d_exp_list)
    params.add('IP', Ne1s, vary=False)
    return params


def tof_photon_energy_model(params, tof, photon_energy=None):
    IP = params['IP'].value

    mod = IP + tof_e_energy_model(params, tof)

    if photon_energy is None:
        return mod
    return mod - photon_energy


def tof_e_beam_energy_params(d_exp_list=[2]):
    params = tof_photon_energy_params(d_exp_list=d_exp_list)
    params.add('K', 2.5)
    return params


def tof_e_beam_energy_model(params, tof, e_beam_energy=None):
    K = params['K'].value
    mod = K * tof_photon_energy_model(params, tof)**2
    if e_beam_energy is None:
        return mod
    return mod - e_beam_energy


def slicing_plot(x, y, z, n_z=6, fig_num=None):
    n_rows = int(np.floor(np.sqrt(n_z)))
    n_cols = int(np.ceil(float(n_z) / n_rows))

    valid = np.isfinite(x) * np.isfinite(y) * np.isfinite(z)
    x_limits = np.linspace(x[valid].min(), x[valid].max(), 2**6)
    y_limits = np.linspace(y[valid].min(), y[valid].max(), 2**6)
    z_limits = np.linspace(z[valid].min(), z[valid].max(), n_z+1)

    fig = plt.figure(fig_num)
    fig.clf()
    ax = None
    for i in range(n_z):
        I = (z_limits[i] < z) & (z < z_limits[i+1])
        img, _, _ = np.histogram2d(x[I], y[I], (x_limits, y_limits))
        ax = fig.add_subplot(n_rows, n_cols, i+1, sharex=ax, sharey=ax)
        ax.imshow(img.T, aspect='auto', interpolation='none',
                  extent=(x_limits[0], x_limits[-1],
                          y_limits[0], y_limits[-1]),
                  origin='lower')
#        ax.plot(x[I], y[I], '.')


def load_tof_to_energy_data(verbose=0):
    if verbose > 0:
        print 'In "load_tof_to_energy_data()".'
    calib_runs = [24, 26, 28, 31, 38]
    calib_energies = [930, 950, 970, 1000, 1030]
    calib_energy_map = dict(zip(calib_runs, calib_energies))

    h5_dict = {}
    for run in calib_runs:
        name = process.h5_file_name_template.format(run)
        process.update_with_noise_and_response(name, verbose=verbose)
        h5_dict[run] = h5py.File(name, 'r+')

    return calib_runs, h5_dict, calib_energy_map


def get_calib_data(plot=False, verbose=0):
    # Load the data
    calib_runs, h5_dict, calib_energy_map = load_tof_to_energy_data(
        verbose=verbose)

    # Make the calib data struct
    calib_data = struct()

    # Make some empty lists
    calib_data.e_energy = []
    calib_data.p_energy_calib = []
    calib_data.p_energy_calib_mean = []
    calib_data.tof = []
    calib_data.integral = []
    calib_data.pulse_energy = []
    calib_data.l3_energy = []
    calib_data.bc2_energy = []
    calib_data.charge = []
#    current_bc2 = []
#    pct = []

    if plot:
        # Make the fee figure
        fee_fig = plt.figure('fee hist')
        fee_fig.clf()
        # Make the energy figure
        energy_fig = plt.figure('energy')
        energy_fig.clf()
        energy_ax_list = []
        # All data should be plotted in the last subplot
        common_ax = energy_fig.add_subplot(2, 3, 6)
        plt.xlabel('L3 energy (MeV)')
        i_plot = 0

    # For each of the runs
    for run, h5 in h5_dict.iteritems():
        # Get the fee values
        fee = h5['fee_mean'].value
        # Check which fee values are actually real
        fee_valid = np.isfinite(fee)
        # Implement a range selection of the valid fee values
        fee_selection = fee_valid & (0.03 < fee) & (fee < 0.08)

        # Fill the fee plot
        if plot:
            # Increment the plot number...
            i_plot += 1
            # ...and make the axis
            ax = fee_fig.add_subplot(2, 3,  i_plot)
            # Make fee histograms
            _, bins, _ = ax.hist(fee[fee_valid], bins=100)
            # and plot the result
            ax.hist(fee[fee_selection], bins=bins)
            ax.set_title('run {}'.format(run))

        # Get the peak center of the fee filtered peaks
        streak_center = h5['streak_peak_center'][fee_selection]
        # Create a selection on the peak center based on the distribution
        sc_mean = streak_center.mean()
        sc_std = streak_center.std()
        streak_selection = (
            (sc_mean - 3*sc_std < h5['streak_peak_center'][:]) &
            (h5['streak_peak_center'][:] < sc_mean + 3*sc_std))

        # Get the corrected l3 energy
        corrected_energy_l3 = h5['energy_L3_corrected_MeV'][fee_selection]
        # and make a similar selection as above
        cor_l3_mean = corrected_energy_l3.mean()
        cor_l3_std = corrected_energy_l3.std()
        cor_l3_selection = (
            (cor_l3_mean - 3*cor_l3_std <
             h5['energy_L3_corrected_MeV'][:]) &
            (h5['energy_L3_corrected_MeV'][:] <
             cor_l3_mean + 3*cor_l3_std))

        # The total shot selection taked all the three above created selections
        # into account
        selection = fee_selection * streak_selection * cor_l3_selection

        # Append the data in the lists
        calib_data.e_energy.append(h5['energy_L3_corrected_MeV'][selection])
        calib_data.l3_energy.append(h5['raw/energy_L3_MeV'][selection])
        calib_data.bc2_energy.append(h5['energy_BC2_MeV'][selection])
        calib_data.p_energy_calib.append(
            [calib_energy_map[run]]*selection.sum())
        calib_data.p_energy_calib_mean.append([calib_energy_map[run]])
        calib_data.tof.append(h5['streak_peak_center'][selection])
        calib_data.integral.append(h5['streak_peak_integral'][selection])
        calib_data.pulse_energy.append(fee[selection])
        calib_data.charge.append(h5['raw/charge_nC'][selection])
#        current_bc2.append(h5['raw/current_BC2_A'][selection])
#        pct.append(h5['raw/phase_cavity_times'][selection, 1])
#        pct[-1] -= pct[-1][np.isfinite(pct[-1])].mean()

        # Populate the energy plot
        if plot:
            # Make the axis
            ax = energy_fig.add_subplot(2, 3, i_plot)
            energy_ax_list.append(ax)
#            plt.plot(energy_l3, streak_center, '.')
            plt.scatter(calib_data.l3_energy[-1],
                        calib_data.tof[-1],
                        s=1, c=calib_data.pulse_energy[-1],
                        linewidths=(0,), alpha=1)
            if i_plot % 3 == 1:
                plt.ylabel('Photoline center (us)')
            if i_plot > 3:
                plt.xlabel('L3 energy (MeV)')

            ax.set_title('run {}'.format(run))
            common_ax.plot(calib_data.l3_energy[-1], calib_data.tof[-1], '.')

    calib_data.tof_mean = [[np.mean(tof_vals)] for tof_vals in calib_data.tof]

    # Convert the data lists to arrays
    if verbose:
        print 'Making data arrays.'
    for k, v in calib_data.toDict().iteritems():
        setattr(calib_data, k, np.concatenate(v))

    return calib_data


def make_tof_to_energy_matrix(energy_scale_eV, plot=False, verbose=0):
    # Get time to energy conversion parameters
    time_to_energy_params, tof_prediction_params = \
        fit_tof_prediction(plot=plot, verbose=verbose)
    # Get the calib data
    calib_data = get_calib_data(plot=plot, verbose=verbose)
    # and unpack the needed parameters
    integral = calib_data.integral
    pulse_energy = calib_data.pulse_energy
    tof = calib_data.tof
#    e_energy = calib_data.e_energy

    # Load the data files
    _, h5_dict, _ = load_tof_to_energy_data()
    # and get the time scale
    time_scale = h5_dict.values()[0]['raw/time_scale'].value

#    e_energy = np.concatenate(e_energy)
#    p_energy_axis = np.linspace(p_energy_calib.min() * 0.99,
#                                p_energy_calib.max() * 1.01,
#                                2**10)
#    tof = np.concatenate(tof)
#    integral = np.concatenate(integral)
#    pulse_energy = np.concatenate(pulse_energy)

#    e_e_axis = np.linspace(4500, 4800, 2**10)
    tof_lims = np.linspace(1.58, 1.66, 2**10 + 1)
    tof_axis = (tof_lims[:-1] + tof_lims[1:]) / 2

    # Convert the effective length to real units (mm)
    D_mm = (time_to_energy_params['d_eff'].value * tof_module.c_0_mps * 1e-3 /
            np.sqrt(tof_module.m_e_eV / 2))
    trans_mat, _ = tof_module.get_time_to_energy_conversion(
        time_scale, energy_scale_eV, verbose=(verbose > 1),
        D_mm=D_mm,
        prompt_us=time_to_energy_params['t_0'].value,
        t_offset_us=0,
        E_offset_eV=time_to_energy_params['E_0'].value)
    trans_mat = trans_mat.toarray()

    if plot:
        plt.figure('trans mat raw')
        plt.clf()
        plt.imshow(trans_mat, interpolation='none', origin='low',
                   aspect='auto', extent=(time_scale.min(), time_scale.max(),
                                          energy_scale_eV.min(),
                                          energy_scale_eV.max()))

    norm_integral = integral/pulse_energy
    binned_norm_int = np.empty_like(tof_axis)
    for i_bin in range(len(binned_norm_int)):
        I = ((tof_lims[i_bin] < tof) &
             (tof < tof_lims[i_bin + 1]) &
             np.isfinite(norm_integral))
        binned_norm_int[i_bin] = norm_integral[I].mean()
    I = np.isfinite(binned_norm_int)
    binned_norm_int = binned_norm_int[I]
    tof_binned_norm_int = tof_axis[I]
    trans_p = np.polyfit(tof, norm_integral, 4)
    trans_model = lmfit.models.SkewedGaussianModel()
    trans_model.set_param_hint('center', value=1.62, min=1.58, max=1.65)
    trans_model.set_param_hint('sigma', value=0.1, min=0, max=1)
    trans_model.set_param_hint('amplitude', value=1, min=0)
    trans_model.set_param_hint('gamma', value=1, min=0)
    trans_params = trans_model.make_params()
    trans_result = trans_model.fit(norm_integral, x=tof,
                                   params=trans_params)

    binned_result = trans_model.fit(binned_norm_int, x=tof_binned_norm_int,
                                    params=trans_params)
    if verbose:
        print '\nBinned fit result:'
        print binned_result.fit_report()
        print '\nRaw data fit result:'
        print trans_result.fit_report()

    if plot:
        plt.figure('intensity')
        plt.clf()
        plt.subplot(131)
        plt.plot(tof, integral, '.', label='direct integral')
        plt.xlabel('tof')
        plt.ylabel('peak integral')

        plt.subplot(132)
        plt.plot(tof, integral/pulse_energy, '.', label='raw')
        plt.plot(tof_axis, np.polyval(trans_p, tof_axis), '-',
                 label='polynomial fit')
        plt.plot(tof_axis,
                 trans_model.eval(x=tof_axis, **trans_result.best_values),
                 label='fit to raw')
        plt.xlabel('tof')
        plt.ylabel('fee normalized peak integral')

        plt.subplot(133)
        plt.plot(tof_binned_norm_int, binned_norm_int, '.',
                 label='binned data')
        plt.plot(tof_axis,
                 trans_model.eval(x=tof_axis, **trans_result.best_values),
                 label='fitt to raw')
        plt.plot(tof_axis,
                 trans_model.eval(x=tof_axis, **binned_result.best_values),
                 label='fit to binned')
        plt.xlabel('tof')
        plt.ylabel('fee normalized peak integral')

    transmission_factors = 1. / trans_model.eval(x=time_scale,
                                                 **binned_result.best_values)
    transmission_factors[(time_scale < tof_lims.min()) |
                         (tof_lims.max() < time_scale)] = 0.0
    transmission_factors[~np.isfinite(transmission_factors)] = 0.0
    transmission_factors /= transmission_factors[
        transmission_factors > 0].min()

    return (trans_mat * transmission_factors,
            time_scale,
            energy_scale_eV,
            time_to_energy_params,
            tof_prediction_params)


def fit_tof_prediction(plot=False, verbose=0):
    if verbose > 0:
        print 'In "fit_tof_prediction()".'
    # Get the calibration data
    calib_data = get_calib_data(plot=plot, verbose=verbose)

    # Unpack calib data
    tof_mean = calib_data.tof_mean
    p_energy_calib_mean = calib_data.p_energy_calib_mean
    l3_energy = calib_data.l3_energy
    bc2_energy = calib_data.bc2_energy
    pulse_energy = calib_data.pulse_energy
    tof = calib_data.tof
    p_energy_calib = calib_data.p_energy_calib

    # Fit the time to energuy conversion using the values given by the
    # operators
    time_to_energy_params = photon_energy_params()
    res = lmfit.minimize(photon_energy_model, time_to_energy_params,
                         args=(tof_mean, p_energy_calib_mean))

    if verbose:
        print 'Time to energy conversion fit results:'
        lmfit.report_fit(res)

    # Select the parameters to be used in the tof time prediction calculation
    var_dict = {'l3_energy': l3_energy,
                'bc2_energy': bc2_energy,
                'fee': pulse_energy,
                'tof': tof
                }

    # Create the parameters for the tof prediction
    prediction_params = tof_prediction_params(**var_dict)
    # Update the parameters from the time to energy conversion
    for k, v in time_to_energy_params.iteritems():
        k_pred = k
        if k != 'IP':
            k_pred += '_prediction'
        prediction_params[k_pred].value = v.value
#        prediction_params[k_pred].vary = False
    # Perform the fit
    res = lmfit.minimize(tof_prediction_model, prediction_params,
                         kws=var_dict)

    # Present the results
    if verbose:
        print 'Tof prediction fit report:'
        lmfit.report_fit(res)

    # Plot the differences between the measured tof and the predicted tof
    if plot:
        time_eps_fig = plt.figure('time eps')
        time_eps_fig.clf()
        plt.scatter(tof,
                    tof_prediction_model(prediction_params,
                                         **var_dict),
                    s=1, c=pulse_energy, linewidths=(0,))
        plt.xlabel('TOF (us)')
        plt.ylabel('TOF prediction error (us)')

    # Look at the time to energy conversion
    if plot:
        time_axis = np.linspace(min(tof), max(tof), 2**8)
        plt.figure('time to energy')
        plt.clf()

        combined_params = lmfit.Parameters()
        for pars in [time_to_energy_params, prediction_params]:
            for k, v in pars.iteritems():
                combined_params.add(k, v.value)

        plt.plot(
            tof,
            photoelectron_energy_prediction_model(combined_params, **var_dict),
            '.', label='prediction + calibration')

        plt.plot(tof, p_energy_calib - Ne1s, '.', label='calibration energies')

        plt.plot(tof_mean, p_energy_calib_mean - Ne1s, 'o',
                 label='calib energies, mean tof')

        plt.plot(time_axis,
                 photoelectron_energy_model(prediction_params,
                                            time_axis,
                                            postfix='_prediction'),
                 label='t -> E tof prediction')

        plt.plot(time_axis,
                 photoelectron_energy_model(time_to_energy_params,
                                            time_axis),
                 label='t -> E calibration')

        plt.xlabel('time (us)')
        plt.ylabel('photo electron energy')
        plt.legend(fontsize='medium')

    return time_to_energy_params, prediction_params


if __name__ == '__main__':
    verbose = 2

#    calib_data = get_calib_data(plot=False)

    energy_scale = np.linspace(40, 160, 2**8)
    (M, time_scale, energy_scale,
     params_time_to_energy,
     params_tof_prediction) = make_tof_to_energy_matrix(
        energy_scale_eV=energy_scale, plot=True, verbose=2)

    h5 = process.load_file('h5_files/run117_all.h5', verbose=1)
    process.list_hdf5_content(h5)
    raw = h5['raw']
    time_scale = raw['time_scale'][:]
    time_signal_dset = h5['filtered_time_signal']
    center_dset = h5['streak_peak_center']
    n_events = len(raw['fiducial'])
    energy_scale = h5['energy_scale_eV'][:]
    energy_signal_dset = h5['energy_signal']
    predicted_energy_dset = h5['photoelectron_energy_prediction_eV']

    selected_shots = list(np.linspace(0, n_events, 10, endpoint=False))
    plt.figure('time_traces')
    plt.clf()
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    ax1.plot(time_scale, time_signal_dset[selected_shots, :].T)
    ax2.plot(np.tile(time_scale, (len(selected_shots), 1)).T -
             center_dset[selected_shots],
             time_signal_dset[selected_shots, :].T)
    ax3.plot(energy_scale, energy_signal_dset[selected_shots, :].T)
    ax4.plot(np.tile(energy_scale, (len(selected_shots), 1)).T -
             predicted_energy_dset[selected_shots],
             energy_signal_dset[selected_shots, :].T)

#    params_time_to_energy, params_tof_prediction = \
#        fit_tof_prediction(plot=True, verbose=verbose)

#    h5 = process.load_file('h5_files/run108_all.h5', verbose=1)
#
#    n_evnts = len(h5['raw/fiducial'])
#    time_scale = h5['raw/time_scale'].value
#    time_signal_dset = h5['filtered_time_signal']
#    time_signal_raw_dset = h5['raw/time_signal']
#    i_evt = np.random.randint(n_evnts)
#    t_trace = time_signal_dset[i_evt, :]
#    t_trace_raw = time_signal_raw_dset[i_evt, :]
#    e_trace = M.dot(t_trace)
#
#    plt.figure('tof to energy conversion test')
#    plt.clf()
#    plt.subplot(121)
#    plt.plot(time_scale, t_trace, label='wiener deconv.')
#    plt.plot(time_scale, t_trace_raw, label='raw')
#    plt.xlabel('tof (us)')
#    plt.legend(loc='best', fontsize='small')
#    plt.subplot(122)
#    plt.plot(energy_scale, M.dot(t_trace))
#    plt.xlabel('Energy (eV)')
    