from pylab import *
ion()
import h5py
import random

f = h5py.File('data/run24_1000.h5', 'r')
print 'Avaliablea data:'
for k in f:
    print '\t{}'.format(k)

N = f.attrs.get('numEvents')
nToPlot = 100
every = N/nToPlot
sl = slice(None, None, every)
EBC2 = f['eBeamEnergyBC2_MeV'][sl]
IBC2 = f['eBeamCurrentBC2_A'][sl]
EL3 = f['eBeamEnergyL3_MeV'][sl]
I = ( EBC2 < 2e5) \
    #& (f['eBeamEnergyL3_MeV'][sl] > 4500) \
    #& (f['eBeamEnergyBC2_MeV'][sl] > 4960) \
#    & (f['eBeamEnergyL3_MeV'][sl] - f['eBeamEnergyBC2_MeV'][sl] > -465) \
#    & (f['eBeamEnergyL3_MeV'][sl] - f['eBeamEnergyBC2_MeV'][sl] < -447)

figure(1); clf()
xlabel('BC2 energy (MeV)')
ylabel('L3 energy (MeV)')
plot(EBC2[I], EL3[I], '.')

figure(2); clf()
xlabel('BC2 energy (MeV)')
ylabel('L3 energy - BC2 energy (MeV)')
plot(EBC2[I], EL3[I] - EBC2[I], '.')


figure(3); clf()
xlabel('BC2 energy (MeV)')
ylabel('BC2 current (A)')
plot(EBC2[I], IBC2[I], '.')


figure(4); clf()
nPlots=4
for i, r in enumerate( random.sample( range(N), nPlots ) ):
    subplot(nPlots,1,i+1)
    plot(f['timeScale_us'], f['rawTimeTrace_V'][r,:])
    plot(f['timeScale_us'], f['deconvTimeTrace_V'][r,:])

f.close()
