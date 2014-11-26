from pylab import *
ion()
import h5py
import os
import wiener


runs = range(132, 136+1)

wf = None
for run in runs:
    f = h5py.File('KrRuns/run{}_all.h5'.format(run), 'r')
    print 'File keys:'
    for k in f.keys():
        print '\t{}'.format(k)

    #figure(run); clf()
    #plot(f['feeEnergy_mJ'][:].mean(axis=1))

    I = f['feeEnergy_mJ'][:].mean(axis=1) > 2e-3

    if wf is None:
        wf = f['rawTimeTrace_V'][:].sum(axis=0)
    else:
        wf += f['rawTimeTrace_V'][:].sum(axis=0)

    t = f['timeScale_us'][:]

    f.close()

#wf -= wf[t<1.5].mean()

figure(1); clf()
plot(t, wf)



I = (1.48 < t) & (t <= 1.5375)
tI = t[I]
wfI = wf[I]

ft = fft(wf)
f = linspace(0, 1e-3/diff(t).mean(), len(t)+1)[:-1]
ftFilt = ft.copy()
ftFilt[ abs(abs(f-4)-2) < 0.001 ] *= 200./6000
filt = abs(ifft(ftFilt))

figure(2); clf()
semilogy(f, abs(ft))
semilogy(f, abs(ftFilt))



figure(3); clf()
plot(tI, wfI, label='average')
plot(tI, filt[I], label='filtered')

legend(loc='best')


offset = filt[t<1.5].mean()
filt -= offset
wf -= offset

figure(5); clf()
plot(tI, wf[I], label='average')

pMin = 1.5115
pMax = 1.533
sPoints = 20
smoothRegion = array( where( (pMin < t) & (t < pMax) ) )
smoothRegion = slice( smoothRegion.min() - sPoints,
        smoothRegion.max() + sPoints )

smoothed = zeros_like(t)
smoothed[smoothRegion] = wiener.edgeSmoothing(filt[smoothRegion] ,
        smoothPoints=sPoints)

plot(tI, smoothed[I], label='filtered and edge-smothed')

legend(loc='best')


response = smoothed
response = response / response.sum()

figure(6); clf()
plot(response)

figure(7); clf()
plot(t, wf)
plot(t, wiener.deconvolution(wf, 10000, response))

with h5py.File('data/KrPrompt.h5', 'w') as f:
    f.create_dataset('timeScale_us', data = t)
    f.create_dataset('responseFunction', data = response)
