import sys
import os
from mpi4py import MPI
import numpy as np
import h5py
import time

import argparse
from aolPyModules import lcls
from aolPyModules import simplepsana

comm = MPI.COMM_WORLD
n_ranks = comm.size
rank = comm.rank

acqiris_source = 'DetInfo(AmoETOF.0:Acqiris.0)'
acqiris_channel = 1
power_meter_source = 'DetInfo(AmoITOF.0:Acqiris.0)'
power_meter_channel = 0
n_bridge = 4
time_range = (1.5, 1.7)

# Some name definitions
lcls_scalars = {'energy_L3_MeV': lcls.getEBeamEnergyL3_MeV,
                'position_BC2_mm': lcls.getEBeamPosOffsetBC2_mm,
                'charge_nC': lcls.getEBeamCharge_nC,
                'current_BC2_A': lcls.getEBeamPkCurrentBC2_A}
if rank == 0:
    lcls_warning_flags = []

# Parse the command line
parser = argparse.ArgumentParser(
        description=('Tool to get data from xtc files into a custom hdf5 '
            + 'format.')
        )

parser.add_argument(
        'dataSource', type = str,
        help=('xtc-file or other description of the data that can be used'
            + ' by psana. Example "exp=amoc8114:run=108". '
            + ':idx will be added as needed.'))

parser.add_argument(
        'hdf5File', type=str,
        help=('Name of hdf5 file to be created.' +
              'The files will be put in /reg/d/psdm/AMO/amoc8114/scratch'))

parser.add_argument(
        '-n', '--numEvents', metavar='N', default = -1, type = int,
        help=('Number of events to process. The events will be distributed'
            + 'over the whole file'))

parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help=('Print stuff to the terminal'))

args =  parser.parse_args()

args.hdf5File = '/reg/d/psdm/AMO/amoc8114/scratch/' + args.hdf5File

# Make a verbose flag
verbose = args.verbose

if verbose:
    print args

# Connect to the data source
if not args.dataSource.endswith(':idx'):
    args.dataSource += ':idx'
if verbose:
    print 'Connecting to the data source "{}"'.format(args.dataSource)
simplepsana.allow_corrupt_epics()
ds = simplepsana.get_data_source(args.dataSource)
run = ds.runs().next()

# Get the time index vector
times = run.times()
events_in_file = len(times)
# Figure out how many events to use
if args.numEvents < 1:
    args.numEvents = np.inf
n_events = min(args.numEvents, events_in_file)

# Figure out whech events shold be handled by which rank
rank_start_event = n_events / n_ranks * rank + min(rank, n_events % n_ranks)
rank_n_events = n_events / n_ranks + (1 if rank < n_events % n_ranks else 0)
rank_stop_event = rank_start_event + rank_n_events
rank_times = times[rank_start_event: rank_stop_event]

#print 'rank', rank, 'has events', rank_start_event, 'to', rank_stop_event,
#print ':', range(rank_start_event, rank_stop_event)

# Get the acqiris time scale
time_scale_full_us = simplepsana.get_acqiris_time_scale_us(run.env(), acqiris_source,
                                                      verbose=False)
# Make the raw time slice
raw_time_slice = slice(time_scale_full_us.searchsorted(max(time_range),
                       side='left'))
time_slice = slice(time_scale_full_us.searchsorted(min(time_range)),
                   time_scale_full_us.searchsorted(max(time_range),
                   side='left'))
bg_end = time_scale_full_us.searchsorted(min(time_range))

bridge_idx_shift = np.arange(time_slice.start, time_slice.start + 5)  % n_bridge

# and time scale
time_scale_us = time_scale_full_us[time_slice]

# and the signal scaling
acq_factor_v, acq_offset_v = simplepsana.get_acqiris_signal_scaling(run.env(),
                                                                    acqiris_source,
                                                                    acqiris_channel,
                                                                    verbose=False)
acq_factor_power_meter_v, acq_offset_power_meter_v = \
        simplepsana.get_acqiris_signal_scaling(run.env(),
                                               power_meter_source,
                                               power_meter_channel,
                                               verbose=False)

# Everyone just grabs an event were there is acqiris data to have something to
# work with
start_time = 0  # initialize the start time
for i_event, t in enumerate(times[: n_events]):
    event = run.event(t)
    if i_event == 0:
        # Get the time of the first event, in whole deconds
        start_time = np.floor(lcls.getEventTime(event))
    try:
        acq_wave = simplepsana.get_acqiris_waveform(event,
                                                    acqiris_source,
                                                    acqiris_channel,
                                                    verbose=False)
        if acq_wave is not None:
            break
    except:
        pass
else:
    print 'No data in requested events.'
    sys.exit()

# Remove any existing file
if rank == 0:
    if os.path.isfile(args.hdf5File):
        os.remove(args.hdf5File)
comm.Barrier()

###################
# Here the ranks have to be in sync (I think)

# Create the hdf5 file in parallel mode
h5_file = h5py.File(args.hdf5File, 'w', driver='mpio', comm=comm)
# Make a group for the raw data
raw_data = h5_file.create_group('raw')
raw_data.attrs.modify('time_stamp', time.time())

# Make space for the raw time scale
raw_data.create_dataset('time_scale', data=time_scale_us)

# Make space for the raw acqiris data
time_signal = raw_data.create_dataset(
        'time_signal', dtype=np.float,
        shape=(n_events, len(acq_wave[time_slice])))

# Make space for the lcls scalars
scalars = {}
for k in lcls_scalars:
    scalars[k] = raw_data.create_dataset(k, dtype=np.float,
                                         shape=(n_events,))

# Make space for the FEE gas detectors
fee = raw_data.create_dataset('FEE_energy_mJ', dtype=np.float,
                              shape=(n_events, 6))

# Make space for the phase cavities
ph_cav = raw_data.create_dataset('phase_cavity_times', dtype=np.float,
                                 shape=(n_events, 2))

# time information
fiducial = raw_data.create_dataset('fiducial', dtype=np.int,
                                   shape=(n_events, ))
event_time = raw_data.create_dataset('event_time_s', dtype=np.float64,
                                     shape = (n_events, ))
raw_data.create_dataset('start_time_s', data=start_time)

# power meter
power_meter = raw_data.create_dataset('power_meter_V', dtype=np.float,
                                      shape=(n_events, ))

###################

#print 'rank', rank, 'has events', rank_start_event, 'to', rank_stop_event,
#print ':', range(rank_start_event, rank_stop_event)

# Get the data
if (rank == 0) and verbose:
    print 'Rank', rank, 'processing', rank_n_events, 'events.'
for i_event, t in zip(range(rank_start_event, rank_stop_event), rank_times):
    if ((rank == 0) and verbose and
            ((i_event-rank_start_event == rank_n_events -1) or
             ((i_event-rank_start_event) / (rank_n_events/30) == 0))):
        progress = 100 * i_event / (rank_n_events - 1)
        print '\r[{:30}] {}%'.format('#'*(progress*30/100), progress),
    event = run.event(t)
    #Acqiris data
    acq_wave = simplepsana.get_acqiris_waveform(
            event,
            acqiris_source,
            acqiris_channel,
            verbose=False)[raw_time_slice].astype(float)

    # Rescale to volts
    acq_wave *= acq_factor_v
    acq_wave -= acq_offset_v
    acq_wave *= -1  # Invert

    # Acalculate bridged offsets
    offsets = [acq_wave[start: bg_end: n_bridge].mean()
               for start in range(n_bridge)]

    # keep only the requested part
    acq_wave_cut = acq_wave[time_slice]
    # Correct the offsets in the bridged channels
    for bridge_idx in range(n_bridge):
        acq_wave_cut[bridge_idx_shift[bridge_idx]: : n_bridge] -= \
                offsets[bridge_idx]

    # Save the raw data
    time_signal[i_event,:] = acq_wave_cut

    # power meter
    power_meter[i_event] = (-acq_offset_power_meter_v +
                               acq_factor_power_meter_v *
                               np.mean(simplepsana.get_acqiris_waveform(
                                   event,
                                   power_meter_source,
                                   power_meter_channel)))

    # LCLS data
    lcls.setEvent(event, verbose=False)
    # Phase cavity timese
    ph_cav[i_event, :] = lcls.getPhaseCavityTimes()
    # Go throught the scalar list
    for name, funk in lcls_scalars.iteritems():
        try:
            scalars[name][i_event] = funk()
        except BaseException as e:
            if name not in lcls_warning_flags:
                print 'Exception "{}" when trying to get {}.'.format(e.message,
                                                                     name)
                lcls_warning_flags.append(name)

    # fee
    fee[i_event, :] = lcls.getPulseEnergy_mJ(nValues=6)

    # time
    fiducial[i_event] = lcls.getEventFiducial()
    event_time[i_event] = lcls.getEventTime(offset=start_time)

print ''
h5_file.close()
