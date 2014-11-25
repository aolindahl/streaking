from pylab import *
ion()
import h5py

f = h5py.File('run24_all.h5', 'r')
print 'Avaliablea data:'
for k in f:
    print '\t{}'.format(k)

every = 100
sl = slice(None, None, every)


figure(1); clf()
xlabel('BC2 energy (MeV)')
ylabel('L3 energy (MeV)')
plot(f['eBeamEnergyBC2_MeV'][sl], f['eBeamEnergyL3_MeV'][sl], '.')

figure(2); clf()
xlabel('BC2 energy (MeV)')
ylabel('L3 energy - BC2 energy (MeV)')
plot(f['eBeamEnergyBC2_MeV'][sl], f['eBeamEnergyL3_MeV'][sl] - f['eBeamEnergyBC2_MeV'][sl], '.')


figure(3); clf()
xlabel('BC2 energy (MeV)')
ylabel('BC2 current (A)')
plot(f['eBeamEnergyBC2_MeV'][sl], f['eBeamCurrentBC2_A'][sl], '.')

f.close()
