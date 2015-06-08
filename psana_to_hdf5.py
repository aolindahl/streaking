import sys
from mpi4py import MPI
import numpy as np
import h5py

import arguments
from aolPyModules import simplepsana

acqiris_source = 'DetInfo(AmoETOF.0:Acqiris.0)'
acqiris_channel = 1
n_bridge = 4
time_range = (0, 100)
bg_end = 1.3

comm = MPI.COMM_WORLD
n_ranks = comm.size
rank = comm.rank

# Parse the command line
args = arguments.parse_cmd_line()
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

# Figure out whech events shole be handled by which rank
rank_start_event = n_events / n_ranks * rank + min(rank, n_events % n_ranks)
rank_n_events = n_events / n_ranks + (1 if rank < n_events % n_ranks else 0)
rank_stop_event = rank_start_event + rank_n_events
rank_times = times[rank_start_event: rank_stop_event]

print 'rank', rank, 'has events', rank_start_event, 'to', rank_stop_event,
print ':', range(rank_start_event, rank_stop_event)

# Get the acqiris time scale
time_scale_us = simplepsana.get_acqiris_time_scale_us(run.env(), acqiris_source,
                                                      verbose=False)
# Make the time slice
time_slice = slice(time_scale_us.searchsorted(min(time_range), side='left'),
                   time_scale_us.searchsorted(max(time_range), side='left'))
# and find the background stop index
bg_end_idx = time_scale_us.searchsorted(bg_end)

# and the signal scaling
acq_factor_v, acq_offset_v = simplepsana.get_acqiris_signal_scaling(run.env(),
                                                                    acqiris_source,
                                                                    acqiris_channel,
                                                                    verbose=False)
# Everyone just grabs an event where ther is acqiris data to have something to
# work with
for t in rank_times:
    try:
        acq_wave = simplepsana.get_acqiris_waveform(run.event(t),
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

###################
# Here the ranks have to be in sync (I think)

# Create the hdf5 file in parallel mode
h5_file = h5py.File(args.hdf5File, 'w', driver='mpio', comm=comm)
# Make a group for the raw data
raw_data = h5_file.create_group('raw')

# Make space for the time scale
raw_data.create_dataset('time_scale', data=time_scale_us[time_slice])

# Make space for the acqiris data
time_signal = raw_data.create_dataset('time_signal', dtype=np.float,
                                      shape=(n_events, len(acq_wave[time_slice])))

###################

print 'rank', rank, 'has events', rank_start_event, 'to', rank_stop_event,
print ':', range(rank_start_event, rank_stop_event)

# Get the data
for i_event, t in zip(range(rank_start_event, rank_stop_event), rank_times):
    #A cqiris data
    acq_wave = simplepsana.get_acqiris_waveform(run.event(t),
                                                acqiris_source,
                                                acqiris_channel,
                                                verbose=False).astype(float)
    # Adjust for different baselines in differne bridged acqiris channels
    for i_start in range(n_bridge):
        acq_wave[i_start::n_bridge] -= acq_wave[i_start:bg_end_idx:n_bridge].mean()
    print 'rank', rank, 'i', i_event, acq_wave.shape, acq_wave



    time_signal[i_event,:] = -(acq_factor_v * acq_wave[time_slice])


h5_file.close()
