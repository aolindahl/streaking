import h5py
import pylab as pl
pl.ion()
from lmfit import minimize, Parameters, report_fit
import sys



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
    mod[t < tP] = nan

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

runs = [24, 26, 28, 31, 38]
energies = {24:930, 26:950, 28:970, 31:1000, 38:1030}


useNEvents = 100

files = {}
slices = {}
events = {}
peakResults = {}


for run in runs:
    files[run] = h5py.File(
            'data/run{}_all.h5'.format(run),
            'r')
    eventsInFile = files[run].attrs.get('numEvents')
    events[run] = min(eventsInFile, useNEvents)
    if eventsInFile > useNEvents:
        step = int(pl.floor( (eventsInFile+0.0)/useNEvents ))
        slices[run] = slice(0, step*useNEvents, step)
        #Is[run] = pl.array( random.sample( range(eventsInFile), useNEvents ) )
        #Is[run].sort()
    else:
        slices[run] = slice(None)


for run in runs:
    pl.figure(run); pl.clf()
    peakResults[run] = pl.zeros((events[run], 3))
    timeScale = files[run]['timeScale_us']
    for i, trace in enumerate(files[run]['deconvTimeTrace_V'][slices[run],:]):
        peakResults[run][i,:] = calcfwhm(timeScale, trace)
        pl.plot(timeScale, trace)

    #for i, index in enumerate(Is[run]):
    #    peakResults[run][i,:] = calcfwhm(files[run]['timeScale_us'], 
    #            files[run]['deconvTimeTrace_V'][index,:])


pl.figure(1); pl.clf()
for s, file in zip(slices.itervalues(), files.itervalues()):
    pl.plot(file['timeScale_us'], file['deconvTimeTrace_V'][s,:].T)
    #plot(file['tof_timeScale_us'], file['tof_timeAmplitudeFiltered_V'][I,:].T /
    #        file['fee_mJ'][I,:].mean(axis=1))



tMax = []
eBeamEnergy = []
eBeamEnergyBC2 = []
eBeamCurrent = []
VMax = []
fwhm = []
for run in runs:
    tMax += [peakResults[run][:,1]]
    VMax += [peakResults[run][:,2]]
    fwhm += [peakResults[run][:,0]]
    eBeamEnergy += [files[run]['eBeamEnergyL3_MeV'][slices[run]].reshape(-1)]
    eBeamEnergyBC2 += [files[run]['eBeamEnergyBC2_MeV'][slices[run]].reshape(-1)]
    eBeamCurrent += [files[run]['eBeamCurrentBC2_A'][slices[run]].reshape(-1)]


# Callibrate the photon energy conversion
eBeamEnergyPerRun = pl.array(eBeamEnergy).mean(axis=1)
eBeamEnergyBC2PerRun = pl.array(eBeamEnergyBC2).mean(axis=1)
eBEREalPerRun = eBeamEnergyPerRun - eBeamEnergyBC2PerRun / 1e3 - 5
photonEnergyPerRun = energies.values()
photonEnergyPerRun.sort()
A = (photonEnergyPerRun/eBEREalPerRun**2).mean()

VMax = pl.concatenate(VMax)
I = VMax > 0.1
VMax = VMax[I]
tMax = pl.concatenate(tMax)[I]
fwhm = pl.concatenate(fwhm)[I]
eBeamEnergy = pl.concatenate(eBeamEnergy)[I]
eBeamEnergyBC2 = pl.concatenate(eBeamEnergyBC2)[I]
#eBeamEnergyBC2 = (eBeamEnergyBC2 - 5e3) / 1e3 + 5e3
eBEReal = eBeamEnergy - eBeamEnergyBC2 + 5e3
eBeamCurrent = pl.concatenate(eBeamCurrent)[I]

alpha = eBeamCurrent - eBeamCurrent.min()
alpha /= alpha.max()

pl.figure(2); pl.clf()
pl.plot(tMax, eBeamEnergy, '.')
pl.plot(tMax, eBEReal, '.')


photonEnergy = A * eBEReal**2
ipNe1s = 870.2

electronEnergy = photonEnergy - ipNe1s

pl.figure(3); pl.clf()
pl.scatter(tMax, electronEnergy, c=eBeamCurrent)

params = startParams()
minimizeOut = minimize(energyFitResiduals, params, args=(tMax, electronEnergy, 1e-6))

t = pl.linspace(1.54, 1.61, 100)
pl.plot(t, energyFitResiduals(params, t))
report_fit(params)


hist, yAx, xAx = pl.histogram2d(electronEnergy, tMax, 500)
pl.figure(4); pl.clf()
pl.imshow(hist, interpolation='none', aspect='auto', origin='lower',
            extent=(xAx.min(), xAx.max(), yAx.min(), yAx.max()))
pl.plot(t, energyFitResiduals(params, t), 'r')

