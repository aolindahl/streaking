import h5py
from pylab import *
ion()
import random
from wavelet_filter import wavelet_filt as wf
import wiener

    
# Grab a fil 
fileName = 'data/run109_all.h5'

# Open it
file = h5py.File(fileName, 'r+')

# Print keys
print 'keys in file are:'
for k in file:
    print '\t',
    print k

N = file.attrs.get('numEvents')

# Make references to some of the data in the h5 file
tAx = file['timeScale_us'][:]
tAmp = file['rawTimeTrace_V'][:] - file['baseline_V'].value

#average the fee over all the four detectors
fee = mean(file['feeEnergy_mJ'], axis=1)



# Make the average of the spectra with signal
tAmpAverage = tAmp.mean(axis=0)

ra = random.randint(0,N)
single = tAmp[ra, :]

wfParams = {'thresh' : 0.05,    # Default None
        'W' : 'db5',            # Default db5
        'levels' : 6}           # Default 6
print wfParams

#singleWf = []
#L = []
#for l in arange(1., 10, 2)/100:
#    wfParams['thresh'] = l
#    print wfParams
#    singleWf.append(wf(single, printTh=True, **wfParams))
#    L.append(l)

singleWf = wf(single, printTh=True, **wfParams)

figure(1); clf()
plot(tAx, tAmpAverage, label='Average')
plot(tAx, single, label='random single shot')
plot(tAx, singleWf, label='random single shot')
#for sWf, l in zip(singleWf, L):
#    plot(tAx, sWf, label='WF {} levels'.format(l))
xlim(1.615, 1.64)
legend(loc='best')


with h5py.File('data/KrPrompt.h5', 'r') as f:
    response = f['responseFunction'][:]

singleDeconv = wiener.deconvolution(single, 1000., response)
singleDeconvWf = wf(singleDeconv, printTh=True)

figure(2); clf()
plot(tAx, single, label='random single shot')
plot(tAx, singleDeconv, label='Deconvoluted')
plot(tAx, singleDeconvWf, label='WF')
legend(loc='best')

file.close()
