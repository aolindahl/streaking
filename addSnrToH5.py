import h5py
from pylab import *
ion()
import random
from lmfit import minimize, Parameters, Parameter, report_errors
from wavelet_filter import wavelet_filt as wf
import wiener

    
# Grab a fil 
fileName = 'data/run108_all.h5'

# Open it
file = h5py.File(fileName, 'r+')

# Print keys
print 'keys in file are:'
for k in file:
    print '\t',
    print k


# Make references to some of the data in the h5 file
tAx = file['timeScale_us'][:]
tAmp = file['rawTimeTrace_V'][:] - file['baseline_V'].value

#average the fee over all the four detectors
fee = mean(file['feeEnergy_mJ'], axis=1)
ebEL3 = file['eBeamEnergyL3_MeV']


# Get the maximum amplitudes in all the treaces
ampMax = tAmp.max(axis=1)

# Look for shots without x-rays.
# Do this by lookig for an amplitude that is sufficiently small
noXray = (ampMax < 0.05) & (fee < 1e-3)

# Average the noise spectra over all the shots without x-rays
# - take fft of each shot
# - take absolute value squared of each shot
# - averag the shots
powerAverageNoise = average(abs(fft(tAmp[noXray,:]))**2, axis=0)
# set the dc components to somethign small
powerAverageNoise[0] = 1e-4


# Make the average of the spectra with signal
powerAverageSignal = mean(abs(fft(tAmp[~noXray,:]))**2, axis=0)
powerSignalAverage = abs(fft(tAmp[~noXray,:].mean(axis=0)))**2
tAmpAverage = tAmp[~noXray,:].mean(axis=0)


# Find the spikes in the distribution spectrum
highPoints = powerAverageNoise > 2e-1
highPoints[0] = True

# Wavelet filter the nose power spectrum
panWT = powerAverageNoise.copy()
panWT[~highPoints] = wf(powerAverageNoise[~highPoints], 0.08, printTh=True)
# Make sure the filtered version is propperly symmetric
panWT[1:] = (panWT[1:] + panWT[-1:0:-1])/2

# Wavelet filter the noise power spectrum
pasWT = powerAverageSignal.copy()
pasWT[~highPoints] = wf(powerAverageSignal[~highPoints], printTh=True)
# Make sure the filtered version is propperly symmetric
pasWT[1:] = (pasWT[1:] + pasWT[-1:0:-1])/2


# make the frequency axis
dt = diff(tAx).mean() * 1e3 # in nanoseconds for frequency in GHz
n = len(tAx)
fAx = linspace(0,1./dt, n+1)[:-1]
df = diff(fAx).mean()


def maskOut(v, mask):
    out = ones_like(v, dtype=bool)
    if len(mask.shape) != 2:
        mask = mask.reshape(-1,2)
    for m in mask:
        out &= ~( (m[0] < v) & (v < m[1]) )
    # symeterizr assuming this is done in the frequency domain on unshifted data
    out[1:] &= out[-1:0:-1]
    return out


maskRegions = array([
    [1.8, 2.2], # 2 GHz stuff broad
    [3.87, 4.01], # 4 GHz stuff broad
    #[1. , 4.1], # Basically remove everything
    #[0.65, 4.1], # even more removed
    #[0.30, 4.1], # even more removed
    [-2, -1]   # dummy region, onlyif no region is unwanted
    ])


signalMask = maskOut(fAx, maskRegions)


noiseSpec = panWT.copy()
noiseSpec[~signalMask] = powerAverageSignal[~signalMask]

signalSpec = sqrt(powerAverageSignal) - sqrt(noiseSpec)
signalSpec[signalSpec < 0] = 0
signalSpec = signalSpec**2

SNR = signalSpec/noiseSpec
G = SNR/(SNR + 1)

#plot the noise spectrum
figure(1); clf()
semilogy(fAx, noiseSpec, label='masked noise')
# Plot the signal spectrum
plot(fAx, signalSpec, label='masked signal-noise')

# Plot the full spectrum from the signal runs
plot(fAx, powerAverageSignal, 'y', label='full spectrum with signal')
plot(fAx, powerSignalAverage, label='full spectrum of average signal')
plot(fAx, pasWT, '--', label='Signal wavelet filtered')

#plot the Wiener filter
plot(fAx, G, label='wiener filter')

legend(loc='best')


xlim(-40*df, fAx.max() + 40*df) 
xlabel('Frequency unshifted (GHz) [Nyquist=4 GHz]')
ylabel('Power')
xlim(xmax=4.01)

y = random.sample(tAmp[~noXray,:], 1)[0]
yF = real(ifft(fft(y) * G))

figure(2); clf()
plot(tAx, y)
plot(tAx, yF)
xlim(1.61, 1.65)

yMean = tAmpAverage
yMeanF = wiener.noise(tAmpAverage, SNR)

figure(3); clf()
plot(tAx, yMean)
plot(tAx, yMeanF)
xlim(1.53, 1.59)
ylim(-0.01, 0.015)

IAuger = (1.53<tAx) & (tAx<1.59)
augerPowerSpectrum = abs(fft(yMean[IAuger]))**2
fAuger = linspace(0, 1e-3/diff(tAx).mean(), IAuger.sum()+1)[:-1]
figure(31); clf()
semilogy(fAuger, augerPowerSpectrum)


#SNR.tofile('SNRrun109.npy')



figure(4); clf()
semilogy(fAx, noiseSpec, label='selected noise spectrum', linewidth=4)
semilogy(fAx, signalSpec, label='signal only', linewidth=4)
semilogy(fAx, powerAverageSignal, '--', label='signal spectrum', linewidth=2)
semilogy(fAx, panWT, '-.', label='noise with wavelet smoothing', linewidth=2)
legend(loc='best')
xlim(xmax=fAx[800])
xlabel('Frequency unshifted (GHz) [Nyquist=4 GHz]')
ylabel('Power')


with h5py.File('data/KrPrompt.h5') as f:
    response = f['responseFunction'][:]

figure(5); clf()
plot(tAx, y)
plot(tAx, wiener.deconvolution(y, SNR))
plot(tAx, wiener.deconvolution(y, SNR, response))
#plot(tAx, wiener.deconvolution(y, 10, response))
#plot(tAx, wiener.deconvolution(y, 100, response))
#plot(tAx, wiener.deconvolution(y, 1000, response))


snrName = 'traceSNR'
if snrName in file:
    if file[snrName].shape != SNR.shape:
        del file[snrName]
snrDataSet = file.require_dataset('traceSNR', shape=SNR.shape, dtype='f')
snrDataSet = SNR

file.close()
