import psana
import numpy as np


EBeamTypeList = (psana.Bld.BldDataEBeamV0,
        psana.Bld.BldDataEBeamV1,
        psana.Bld.BldDataEBeamV2,
        psana.Bld.BldDataEBeamV3,
        psana.Bld.BldDataEBeamV4,
        psana.Bld.BldDataEBeamV5,
        psana.Bld.BldDataEBeamV6)

_EBeamType = None
_EBeamSource = psana.Source('BldInfo(EBeam)')

feeTypeList = (psana.Bld.BldDataFEEGasDetEnergy,
        psana.Bld.BldDataFEEGasDetEnergyV1)

_feeType = None
_feeSource = psana.Source('BldInfo(FEEGasDetEnergy)')

def getEBeamEnergyL3_MeV(evt, verbose=False):
    EBeamObject = getEBeamObject(evt, verbose)
    if EBeamObject == None:
        return np.nan
    return EBeamObject.ebeamL3Energy()

def getEBeamEnergyBC2_MeV(evt, verbose=False):
    EBeamObject = getEBeamObject(evt, verbose=verbose)
    if EBeamObject == None:
        return np.nan
    # The BC2 energy is calculated using the dispersion. The vispersion
    # value for BC2 is -3.647 mm at the nominal beam energy of 5 GeV [Email
    # from Timothy Maxwell to Anton Lindahl on June 2 2014]
    return ( EBeamObject.ebeamEnergyBC2() / -364.7 + 1 ) * 5e3

def getEBeamCharge_nC(evt, verbose=False):
    EBeamObject = getEBeamObject(evt, verbose)
    if EBeamObject == None:
        return np.nan
    return EBeamObject.ebeamCharge()

def getEBeamPkCurrentBC2_A(evt, verbose=False):
    EBeamObject = getEBeamObject(evt, verbose)
    if EBeamObject == None:
        return np.nan
    return EBeamObject.ebeamPkCurrBC2()

def getEBeamObject(evt, verbose=False):
    # Initialize the EBeam type
    if _EBeamType is None:
        _determineEBeamType(evt, verbose=verbose)
    return evt.get(_EBeamType, _EBeamSource)

def _determineEBeamType(evt, verbose=False):
    global _EBeamType
    if verbose:
        print 'Find the correct EBeam type.'
    for type in EBeamTypeList:
        if verbose:
            print 'Trying {};'.format(type),
        data = evt.get(type, _EBeamSource)
        if data is not None:
            _EBeamType = type
            if verbose:
                print ' correct.'
            break
        elif verbose:
            print ' wrong one.'

def getPulseEnergy_mJ(evt):
    fee = getFeeObject(evt)
    if fee is None:
        return np.array([np.nan for i in range(4)])
    return np.array( [ fee.f_11_ENRC(),
            fee.f_12_ENRC(),
            fee.f_21_ENRC(),
            fee.f_22_ENRC()] )

def getFeeObject(evt, verbose=False):
    if _feeType is None:
        _determineFeeType(evt, verbose)
    return evt.get(_feeType, _feeSource)

def _determineFeeType(evt, verbose=False):
    global _feeType
    if verbose:
        print 'Find the correct fee type.'
    for type in feeTypeList:
        if verbose:
            print 'Trying {};'.format(type),
        data = evt.get(type, _feeSource)
        if data is not None:
            _feeType = type
            if verbose:
                print ' correct.'
            break
        elif verbose:
            print ' wrong one.'

if __name__ == '__main__':
    ds = psana.DataSource('exp=amoc8114:run=24')
    evt = ds.events().next()
    print 'E at L3 is {} MeV'.format(getEBeamEnergyL3_MeV(evt,
        verbose=True))
    print 'E at BC2 is {} MeV'.format(getEBeamEnergyBC2_MeV(evt,
        verbose=True))
    print 'Q is {} nC'.format(getEBeamCharge_nC(evt,
        verbose=True))
    print 'I is {} A'.format(getEBeamPkCurrentBC2_A(evt,
        verbose=True))
    print 'fee is {} mJ'.format(getPulseEnergy_mJ(evt))

