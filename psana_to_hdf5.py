"""
Extract data from LCLS xtc files into hdf5 files.

Runs as a script and takes command line arguments.
"""

import sys
import os
from mpi4py import MPI
import numpy as np
import h5py
import time

import argparse
from aolPyModules import lcls
from aolPyModules import simplepsana

# Set up some MPI parameters
comm = MPI.COMM_WORLD
n_ranks = comm.size
rank = comm.rank

# Define some parameters used in the analysis
acqiris_source = 'DetInfo(AmoETOF.0:Acqiris.0)'
acqiris_channel = 1
power_meter_source = 'DetInfo(AmoITOF.0:Acqiris.0)'
power_meter_channel = 0
n_bridge = 4  # The number of bridged channels in the acqiris board.
time_range = (1.5, 1.7)  # Signal ROI.

# Some name definitions used to handle scalar values in the data.
# The notation defines the name to be used for the datasetd in the raw data
# group as key and the function called to get the scalar as value in the
# dictionary.
lcls_scalars = {'energy_L3_MeV': lcls.getEBeamEnergyL3_MeV,
                'position_BC2_mm': lcls.getEBeamPosOffsetBC2_mm,
                'charge_nC': lcls.getEBeamCharge_nC,
                'current_BC2_A': lcls.getEBeamPkCurrentBC2_A}

# The master keeps a list of warning flags
if rank == 0:
    lcls_warning_flags = []

# Parse the command line
parser = argparse.ArgumentParser(
        description=('Tool to get data from xtc files into a custom hdf5 ' +
                     'format.'))

parser.add_argument(
        'dataSource', type=str,
        help=('xtc-file or other description of the data that can be used' +
              ' by psana. Example "exp=amoc8114:run=108".' +
              ' :idx will be added as needed.'))

parser.add_argument(
        'hdf5File', type=str,
        help=('Name of hdf5 file to be created.' +
              'The files will be put in /reg/d/psdm/AMO/amoc8114/scratch'))

parser.add_argument(
        '-n', '--numEvents', metavar='N', default=-1, type=int,
        help=('Number of events to process. The events will be distributed' +
              'over the whole file'))

parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help=('Print stuff to the terminal'))

args = parser.parse_args()

args.hdf5File = '/reg/d/psdm/AMO/amoc8114/scratch/' + args.hdf5File

# Make a verbose flag
verbose = args.verbose

if verbose:
    print args

# Make sure the dat source ends with ':idx'.
if not args.dataSource.endswith(':idx'):
    args.dataSource += ':idx'  # Add :idx if necessary
if verbose:
    print 'Connecting to the data source "{}"'.format(args.dataSource)
# With the used sampling rate (I think) there is a prblem with the indexing of
# the xtc files. I'm not sure if this causes problems, but in order to run
# using MPI this is needed.
simplepsana.allow_corrupt_epics()
# Connect to the data source
ds = simplepsana.get_data_source(args.dataSource)
# This is set up to only handle one run at the time althought the psana data
# source specification can define multiple runs. First run is used here.
run = ds.runs().next()

# Get the time index vector
times = run.times()
events_in_file = len(times)
# Figure out how many events to use
if args.numEvents < 1:  # < 1 gives all events in the run
    args.numEvents = np.inf
n_events = min(args.numEvents, events_in_file)

# Figure out whech events shold be handled by which rank.
rank_start_event = n_events / n_ranks * rank + min(rank, n_events % n_ranks)
rank_n_events = n_events / n_ranks + (1 if rank < n_events % n_ranks else 0)
rank_stop_event = rank_start_event + rank_n_events
rank_times = times[rank_start_event: rank_stop_event]

#print 'rank', rank, 'has events', rank_start_event, 'to', rank_stop_event,
#print ':', range(rank_start_event, rank_stop_event)

# Get the full acqiris time scale.
time_scale_full_us = simplepsana.get_acqiris_time_scale_us(run.env(),
                                                           acqiris_source,
                                                           verbose=False)

# Make the raw time slice, including all data up to the end of the set ROI
# defined in 'time_range'.
raw_time_slice = slice(time_scale_full_us.searchsorted(max(time_range),
                                                       side='left'))
# Define the time slice for the ROI.
time_slice = slice(time_scale_full_us.searchsorted(min(time_range)),
                   time_scale_full_us.searchsorted(max(time_range),
                                                   side='left'))
# Use the start of the ROI to find the index where the background ends.
bg_end = time_scale_full_us.searchsorted(min(time_range))

# Shifts connected to the bridged channels in the acqiris compared to the start
# index in the time slice vector.
bridge_idx_shift = np.arange(time_slice.start,
                             time_slice.start + n_bridge) % n_bridge

# Get the sliced time scale for the ROI.
time_scale_us = time_scale_full_us[time_slice]

# Determinie the sinal scaling for the acqiris used for the TOF.
acq_factor_v, acq_offset_v = simplepsana.get_acqiris_signal_scaling(
    run.env(), acqiris_source, acqiris_channel, verbose=False)
# And the dcaling for the channel used for the power meter.
acq_factor_power_meter_v, acq_offset_power_meter_v = \
        simplepsana.get_acqiris_signal_scaling(run.env(),
                                               power_meter_source,
                                               power_meter_channel,
                                               verbose=False)

# Every rank just grabs an event were there is acqiris data to have something
# to work with in the setup stage.
# Iterate over (potentially) all the event times that are defined to be used.
for i_event, t in enumerate(times[: n_events]):
    event = run.event(t)  # Get the corresponding event.
    if i_event == 0:
        # Get the time of the first event, in whole seconds only.
        start_time = np.floor(lcls.getEventTime(event))
    try:
        # Try grabbing the acqiris waveform for the TOF.
        acq_wave = simplepsana.get_acqiris_waveform(event,
                                                    acqiris_source,
                                                    acqiris_channel,
                                                    verbose=False)
        if acq_wave is not None:
            # Something else that None means we are happy so break out of the
            # for loop over the events.
            break
    except:
        # Failiure can occure if there is no data in the event.
        # Anyhow, just skip to the next iteration of the for loop.
        pass
else:
    # If the for loop runs through all the data without break there is truely
    # no data in the specified range.
    print 'No data in requested events.'
    sys.exit()

# Only rank 0 should attempt this.
if rank == 0:
    # Remove any existing file with the specified name.
    if os.path.isfile(args.hdf5File):
        os.remove(args.hdf5File)

# Wait untill all ranks are at this point.
comm.Barrier()

###################
# Here the ranks have to be in sync (I think).

# Create the hdf5 file in parallel mode.
h5_file = h5py.File(args.hdf5File, 'w', driver='mpio', comm=comm)
# Make a group for the raw data.
raw_data = h5_file.create_group('raw')
# Time stamp the raw group with the current time.
raw_data.attrs.modify('time_stamp', time.time())

# Write the raw time scale to file.
raw_data.create_dataset('time_scale', data=time_scale_us)

# Make space for the raw acqiris data for the specified number of events.
time_signal = raw_data.create_dataset(
        'time_signal', dtype=np.float,
        shape=(n_events, len(acq_wave[time_slice])))

# Create an empty dictionary for the data sets conected to the lcls scalars.
lcls_scalar_datasets = {}
for k in lcls_scalars:
    # Add a dataset with a single float for each event to the dictionary.
    lcls_scalar_datasets[k] = raw_data.create_dataset(
        k, dtype=np.float, shape=(n_events,))

# Make space for the FEE gas detectors.
fee = raw_data.create_dataset('FEE_energy_mJ', dtype=np.float,
                              shape=(n_events, 6))

# Make space for the phase cavities.
ph_cav = raw_data.create_dataset('phase_cavity_times', dtype=np.float,
                                 shape=(n_events, 2))

# Make space for timing information.
fiducial = raw_data.create_dataset('fiducial',
                                   dtype=np.int,
                                   shape=(n_events, ))
event_time = raw_data.create_dataset('event_time_s',
                                     dtype=np.float64,
                                     shape=(n_events, ))
# Save the start time to file.
raw_data.create_dataset('start_time_s', data=start_time)

# Make the power meter data set.
power_meter = raw_data.create_dataset('power_meter_V',
                                      dtype=np.float,
                                      shape=(n_events, ))

###################

#print 'rank', rank, 'has events', rank_start_event, 'to', rank_stop_event,
#print ':', range(rank_start_event, rank_stop_event)

# Rank 0 will show a progress bar (if verbose is set).
if (rank == 0):
    # The progress bar update interval in events is given by.
    progress_step = max(1, rank_n_events/100)
    if verbose:
        # Also print a status message.
        print 'Rank', rank, 'processing', rank_n_events, 'events.'

# Get the data.
# Each rank interates over the events assigned to it.
for i_event, t in zip(range(rank_start_event, rank_stop_event), rank_times):
    # Rank 0 should update the progress bar if verbose...
    if ((rank == 0) and verbose and (
            # ...if the current event number is a multiple of the update
            # interval...
            (i_event % progress_step == 0) or
            # ... and for the last of its events in order to always end att
            # 100 %.
            (i_event == rank_stop_event - 1))):
        # So, update the progress bar!
        progress = 100 * (i_event-rank_start_event) / (rank_n_events - 1)
        print '\r[{:30}] {}%'.format('#'*(progress*30/100), progress),
        sys.stdout.flush()

    # Here everyone is in, grab the envet corresponding to the rank specific
    # event time.
    event = run.event(t)

    ######
    # Get the acqiris waveform for the TOF in the event. Directly slice it
    # using the raw_time_slice and convert from int to float.
    acq_wave = simplepsana.get_acqiris_waveform(
            event,
            acqiris_source,
            acqiris_channel,
            verbose=False)[raw_time_slice].astype(float)

    # Rescale the waveform to volts.
    acq_wave *= acq_factor_v
    acq_wave -= acq_offset_v
    acq_wave *= -1  # Invert

    # Calculate the offsets for each of the bridged channels in the region
    # before the defined ROI.
    offsets = [acq_wave[start: bg_end: n_bridge].mean()
               for start in range(n_bridge)]

    # Now the data in front of the ROI is thrown away.
    acq_wave_cut = acq_wave[time_slice]

    # Correct the offsets in the bridged channels
    for bridge_idx in range(n_bridge):
        acq_wave_cut[bridge_idx_shift[bridge_idx]:: n_bridge] -= \
            offsets[bridge_idx]

    # Save the raw data with the offset correction.
    time_signal[i_event, :] = acq_wave_cut

    ######
    # Store the power meter data. In one go the waveform is extracted, averaged
    # and converted to a voltage scale and stored in the hdf5 file.
    power_meter[i_event] = (-acq_offset_power_meter_v +
                            acq_factor_power_meter_v *
                            np.mean(simplepsana.get_acqiris_waveform(
                                    event,
                                    power_meter_source,
                                    power_meter_channel)))

    ######
    # Handle the lcls machine data.
    # First set the current event in the lcls module.
    lcls.setEvent(event, verbose=False)

    # Extract and store the phase cavity timese.
    ph_cav[i_event, :] = lcls.getPhaseCavityTimes()

    # Go throught the scalar list.
    for name, funk in lcls_scalars.iteritems():
        try:
            # Try to get and store the corresponding value.
            lcls_scalar_datasets[name][i_event] = funk()
        except BaseException as e:
            # If this causes problems, add the warning flag to the list of
            # warnings. Only rank 0. And print the message.
            if (rank == 0) and (name not in lcls_warning_flags):
                print 'Exception "{}" when trying to get {}.'.format(e.message,
                                                                     name)
                lcls_warning_flags.append(name)

    # Extract and store the fee data
    fee[i_event, :] = lcls.getPulseEnergy_mJ(nValues=6)

    # Extract and store the timing information.
    fiducial[i_event] = lcls.getEventFiducial()
    event_time[i_event] = lcls.getEventTime(offset=start_time)

if rank == 0:
    print ''

# Close the hdf5 file.
h5_file.close()
