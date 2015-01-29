import h5py
import pylab as pl
pl.ion()
from lmfit import minimize, Parameters, report_fit
#import sys



def calcfwhm(x,y, fraction=0.5):
    """
    calcfwhm(x,y) - with input x,y vector this function calculate fwhm
                    and return (fwhm,xpeak,ymax)
    x - input independent variable
    y - input dependent variable
    fwhm - return full width half maximum
    xpeak - return x value at y = ymax
    """
    ymax = max(y)
    ymin = min(y)
    y_hpeak = ymin + fraction *(ymax-ymin)
    x_hpeak = []
    NPT = len(x)
    for i in range(NPT):
        if y[i] >= y_hpeak:
            i1 = i
            break
    for i in range(i1+1,NPT):
        if y[i] <= y_hpeak:
            i2 = i
            break
        if i == NPT-1: i2 = i
    if y[i1] == y_hpeak: x_hpeak_l = x[i1]
    else:
        x_hpeak_l = (y_hpeak-y[i1-1])/(y[i1]-y[i1-1])*(x[i1]-x[i1-1])+x[i1-1]
    if y[i2] == y_hpeak: x_hpeak_r = x[i2]
    else:
        x_hpeak_r = (y_hpeak-y[i2-1])/(y[i2]-y[i2-1])*(x[i2]-x[i2-1])+x[i2-1]
    x_hpeak = [x_hpeak_l,x_hpeak_r]

    fwhm = abs(x_hpeak[1]-x_hpeak[0])
    for i in range(NPT):
        if y[i] == ymax:
            jmax = i
            break
    xpeak = x[jmax]
    return (fwhm,xpeak,ymax)


def energyFitResiduals(params, t, E=None, eps=None):
    # E = D^2 m_e 10^6 / ( 2 c_0^2 (t - t_p)^2 ) + E0,
    # where:
    # D is the flight distance in mm
    # m_e is the electon rest mass expressed in eV
    # the 10^6 factor accounts the the otherwhise missmachching prefixes
    # c_0 is the speed of light in m/s
    # E is the energy in eV
    # E0 is an energy offset in eV, should be determined in a callibration
    # fit
    # t_p is the arrival time of the prompt signal in microseconds

    m_e_eV = 0.510998928e6 # http://physics.nist.gov/cgi-bin/cuu/Value?me 2014-04-21
    c_0_mps = 299792458 # http://physics.nist.gov/cgi-bin/cuu/Value?c|search_for=universal_in! 2014-04-21

    # the parameters
    D = params['D'].value
    tP = params['tP'].value
    E0 = params['E0'].value

    mod = (D**2 * m_e_eV * 1e6 / (c_0_mps**2 * 2 * (t - tP)**2) + E0)
    mod[t < tP] = pl.nan

    if E==None:
        return mod
    if eps==None:
        return  mod -E
    return (mod - E) * eps**-2

def startParams():
    params = Parameters()
    params.add('D', value=600)
    params.add('tP', value=1.5)
    params.add('E0', value=0)
    return params



####################

# List of the callibration runs
runs = [24, 26, 28, 31, 38]
# Run number to set photon energy lookup
energies = {24:930, 26:950, 28:970, 31:1000, 38:1030}

# The number of events from each run to use
useNEvents = 1000

# Some empty dictionaries for:
# the hdf5 files
files = {}
# the event slices for the runs
slices = {}
# the number of events to use in a given file
events = {}
# peak finder results
peakResults = {}

# Make some empty lists
tMax = []
eBeamEnergyL3 = []
eBeamEnergyBC2 = []
eBeamCurrent = []
VMax = []
fwhm = []


# Prepare a plot for all traces
traceFig = pl.figure('All traces');
traceFig.clf()
traceAx = traceFig.add_subplot(111)

# Iterate over all the runs
for run in runs:
    # Open the corresponding file and put it in the dictionary
    print 'Processing run {}.'.format(run)
    files[run] = h5py.File(
            'data/run{}_all.h5'.format(run),
            'r')
    # Get the nubmer of events in the file
    eventsInFile = files[run].attrs.get('numEvents')
    # The number of events to use is all in the file or limited by useNEvents
    events[run] = min(eventsInFile, useNEvents)
    # If a limitation is set construct a slice object for some sort of evenly
    # distributed sampling of the whole file
    if eventsInFile > useNEvents:
        # stride length is given by the largest number such that the correct
        # number of strides is still confined in the file
        step = int(pl.floor( (eventsInFile+0.0)/useNEvents ))
        # The run specific slice is then given by the stride and the number of
        # events requested
        slices[run] = slice(0, step*useNEvents, step)
    else:
        # With no event restriction, make an empty slice (include all)
        slices[run] = slice(None)


    # Prepare a plot for the traces in the run
    runTraceFig = pl.figure('Traces in run {}'.format(run))
    runTraceFig.clf()
    runTraceAx = runTraceFig.add_subplot(111)
    # Figure out how often to plot (for 100 traces)
    plotStride = events[run]/100
    # Make space for the peak finder results for the current run
    peakResults[run] = pl.zeros((events[run], 3))
    # get the time scale vector
    timeScale = files[run]['timeScale_us']
    # Iterate over all the selected deconvolutes time traces
    for i, trace in enumerate(files[run]['deconvTimeTrace_V'][slices[run],:]):
        # Do peak finding and store the results
        peakResults[run][i,:] = calcfwhm(timeScale, trace)
        # Plot selected traces
        if i%plotStride == 0:
            runTraceAx.plot(timeScale, trace)
            traceAx.plot(timeScale, trace)



#pl.figure(1); pl.clf()
#for s, file in zip(slices.itervalues(), files.itervalues()):
#    pl.plot(file['timeScale_us'], file['deconvTimeTrace_V'][s,:].T)
    #plot(file['tof_timeScale_us'], file['tof_timeAmplitudeFiltered_V'][I,:].T /
    #        file['fee_mJ'][I,:].mean(axis=1))



    # Add data to the lists.
    # The data is added as an array for each run
    tMax += [peakResults[run][:,1]] # Peak time
    VMax += [peakResults[run][:,2]] # Peak voltage
    fwhm += [peakResults[run][:,0]] # fwhm of the peak
    # e-beam energy at L3
    eBeamEnergyL3 += [files[run]['eBeamEnergyL3_MeV'][slices[run]].reshape(-1)]
    # e-beam energy at BC2
    eBeamEnergyBC2 += [files[run]['eBeamEnergyBC2_MeV'][slices[run]].reshape(-1)]
    # e-beam current at BC2
    eBeamCurrent += [files[run]['eBeamCurrentBC2_A'][slices[run]].reshape(-1)]


# Callibrate the photon energy conversion
# Average over the runs to get representative data for the different settings
eBeamEnergyL3PerRun = pl.array(eBeamEnergyL3).mean(axis=1)
eBeamEnergyBC2PerRun = pl.array(eBeamEnergyBC2).mean(axis=1)
# Calculate the real e-beam energy corresponding to the unspoild part of the
# electron bunch
# This is done by adding to the L3 energy the difference between the measured
# e-beam energy at BC2 that exceeds the nominal 5.0 GeV
eBeamEnergyRealPerRun = eBeamEnergyL3PerRun + (eBeamEnergyBC2PerRun - 5.0e3)
# Get the set photon energies form the table
photonEnergyPerRun = energies.values()
# In the simplest picture the photon energty scales as e-beam energy squared.
# Calculate a scaling factor that somehow represents all the data.
A = (photonEnergyPerRun/eBeamEnergyRealPerRun**2).mean()


# Make single arrays of all the collectes shot data
VMax = pl.concatenate(VMax)
# Put a threshold on the signal level and create an indexing vector
I = VMax > 0.1
# Use it to index all the single shot data
VMax = VMax[I]
tMax = pl.concatenate(tMax)[I]
fwhm = pl.concatenate(fwhm)[I]
eBeamEnergyL3 = pl.concatenate(eBeamEnergyL3)[I]
eBeamEnergyBC2 = pl.concatenate(eBeamEnergyBC2)[I]
# Calculate the real e-beam energy as above for the run averaged data
eBeamEnergyReal = eBeamEnergyL3 - (eBeamEnergyBC2 - 5.0e3)
eBeamCurrent = pl.concatenate(eBeamCurrent)[I]


# Not sure wehat this is... 2015-01-29
alpha = eBeamCurrent - eBeamCurrent.min()
alpha /= alpha.max()

# Make a plot of the e-beam energies vs time. Raw and corrected
pl.figure('Energy vs Tmie'); pl.clf()
pl.plot(tMax, eBeamEnergyL3, '.', label='Raw L3 energy')
pl.plot(tMax, eBeamEnergyReal, '.', label='BC2 corrected energy')
pl.xlabel('Time (us)')
pl.ylabel('e-beam energy (MeV)')
pl.legend(loc='best')

# Using the above calculated scaling factor calculate photon energies for all
# the shots
photonEnergy = A * eBeamEnergyReal**2
# Ionization potential of Ne 1s
ipNe1s = 870.2
# IP and photon energy gives electron energy
electronEnergy = photonEnergy - ipNe1s

# Make a photo electron energy figure
pl.figure('Electron energy'); pl.clf()
pl.scatter(tMax, electronEnergy, c=eBeamCurrent)
pl.xlabel('Time (us)')
pl.ylabel('Electron energy (eV)')

# Make a fit to the electron energy data
# Get the start parameters
params = startParams()
# do the fit.
minimizeOut = minimize(energyFitResiduals, params,
                       args=(tMax, electronEnergy))
# Make a time axis
t = pl.linspace(1.53, 1.59, 100)
# add the fit result to the plot
pl.plot(t, energyFitResiduals(params, t))
# Print the parameters
report_fit(params)


hist, yAx, xAx = pl.histogram2d(electronEnergy, tMax, 500)
pl.figure('histogram'); pl.clf()
pl.imshow(hist, interpolation='none', aspect='auto', origin='lower',
            extent=(xAx.min(), xAx.max(), yAx.min(), yAx.max()))
pl.plot(t, energyFitResiduals(params, t), 'r')

