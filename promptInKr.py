import pylab as pl
pl.ion()
import h5py
import sys
sys.path.append('aolPyModules/')
import wiener

# Plot flag to controll the display behaviour
doPlot = False

# The Kr runs
runs = [132, 133, 134, 135, 136]
#runs = [132]

# Start with a waveform that is just Nonen
wf = None
# and a time axis that is None
t = None
# Loop over the runs
for run in runs:
    # Open the corresponding file
    with h5py.File('data/KrRuns/run{}_all.h5'.format(run), 'r') as f:
        # Print some information about the file and the content
        print 'File {} opened,'.format(f.filename)
        print 'it has the keys:'
        for k in f.keys():
            print '\t{},\tshape{}'.format(k, f[k].shape)


        # Get the time vector
        tTemp = f['timeScale_us'][:]
        if t is None:
            # keep it if it is the first one
            t = tTemp.copy()
        # Check that the new one is the same as the old one
        elif (t.shape != tTemp.shape) and (any(t != tTemp)):
            print 'ERROR: The time axies in the files are not the same. Exit.'
            raise Exception('Current time axis not the same as saved one.')

        # Find the shots with high gas detector energy
        feeMean = f['feeEnergy_mJ'][:].mean(axis=1)
        shotNumber = pl.arange(len(feeMean))
        I = feeMean > 0.002

        if doPlot:
            pl.figure('FEE run {}'.format(run)); pl.clf()
            pl.plot(feeMean, '.', label='All shots')
            pl.plot(shotNumber[I], feeMean[I], '.',
                    label='Selected shots shots')
            pl.xlabel('Shot number')
            pl.ylabel('FEE mean (mJ)')
            pl.legend(loc='best')

        if doPlot:
            pl.figure('Trace mean run {}'.format(run)); pl.clf()
            # Mean of all shots
            pl.plot(t, f['rawTimeTrace_V'][:].mean(axis=0))
            # Mean of high energy shots
            if len(I) > 0:
                pl.plot(t, f['rawTimeTrace_V'][:][I,:].mean(axis=0))
            pl.xlabel('Time (us)')
            pl.ylabel('Signal (V)')

        # Check if the waveform is initialized.
        if wf is None:
            # If not
            # Get the sum of the traces with high and assign to the waveform
            wf = f['rawTimeTrace_V'][:][I,:].sum(axis=0)
        else:
            # If initilaiizwd
            # Add the sum of the new data to the old
            wf += f['rawTimeTrace_V'][:][I,:].sum(axis=0)



#wf -= wf[t<1.5].mean()

# Plot the full waveform
pl.figure('Total waveform'); pl.clf()
pl.plot(t, wf)
pl.xlabel('Time (us)')
pl.ylabel('Signal (V)')

# Define the region of interest for the prompt
I = (1.48 < t) & (t <= 1.5375)
# Add to plot
pl.plot(t[I], wf[I], 'r', label='Prompt ROI')
pl.legend(loc='best')


# Goal here is to filter out the 2 GHz noise
# Fft of the full trace
ft = pl.fft(wf)
# Make the frequency axis
f = pl.linspace(0, 1e-3/pl.diff(t).mean(), len(t)+1)[:-1]
# Make a copy for filtering
ftFilt = ft.copy()
# Attenuate the components at 2 and -2 GHz
ftFilt[ abs(abs(f-4)-2) < 0.001 ] *= 200./6000
# Filtered waveform
filt = abs(pl.ifft(ftFilt))

# Determine the offset of the signal using times before 1.5 us
offset = filt[t<1.5].mean()
# shift the raw and filtered traces correspondingly
filt -= offset
wf -= offset

pl.figure('Frequency domain'); pl.clf()
pl.semilogy(f, abs(ft), label='raw')
pl.semilogy(f, abs(ftFilt), label='filtered')
pl.xlabel('Frequency (GHz)')
pl.ylabel('Spectral power')
pl.legend(loc='best')



pl.figure('Prompt in time domain'); pl.clf()
pl.plot(t[I], wf[I], label='raw')
pl.plot(t[I], filt[I], label='filtered')
pl.xlabel('Time (us)')
pl.ylabel('Signal (V)')

# Define the edge smoothing of the prompt signal
# Times give the unaffected region
pMin = 1.5115 # us
pMax = 1.533 # us
# Smoothin region outside the above time span
sPoints = 20 # number of points
smoothRegion = slice( t.searchsorted(pMin, side='right') - sPoints,
                     t.searchsorted(pMax, side='left') + sPoints)

# initialize the smoothed vector with zeros over the whole interval
smoothed = pl.zeros_like(t)
# In the smoothed region take the raw filtered data and push the edges to zero
# using sin^2 functions.
smoothed[smoothRegion] = wiener.edgeSmoothing(filt[smoothRegion],
        smoothPoints=sPoints)

pl.plot(t[I], smoothed[I], label='filtered and edge-smothed')

pl.legend(loc='best')

# The response function should be normalized so that the sum of the point = 1
response = smoothed / smoothed.sum()

pl.figure('Response'); pl.clf()
pl.plot(t, response)
pl.xlabel('Time (us)')
pl.ylabel('Signal (V)')


# As a test attempt a deconvolution of the full trace using the response
pl.figure('Deconvolution'); pl.clf()
# Just a simple fft-ifft deconvolution
pl.plot(t, pl.ifft( pl.fft(wf) / pl.fft(response)), label='raw devconvolution')
# Deconvolution using a wiener deconvolution and a given signal to noise ratio
# a rather blunt attempt to reduce the influence of the high gain at high
# frequencies.
pl.plot(t, wiener.deconvolution(wf, 10000, response), label='"damped" deconconv.')
# The raw data
pl.plot(t, wf, label='raw')
pl.legend(loc='best')
pl.xlabel('Time (us)')
pl.ylabel('Signal (V)')

responseFile = 'data/KrPrompt.h5'
print 'Overwriting any existing file named "{}" with the prompt data'.format(
    responseFile)
with h5py.File(responseFile, 'w') as f:
    f.create_dataset('timeScale_us', data = t)
    f.create_dataset('responseFunction', data = response)

# Plot file content jus to check
print 'Reading file and plotting content, just to check.'
with h5py.File(responseFile, 'r') as f:
    pl.figure('File content'); pl.clf()
    pl.plot(f['timeScale_us'], f['responseFunction'])
print 'Done'