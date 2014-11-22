import psana
import numpy as np
from scipy.sparse import coo_matrix

segment = 0 # Somehow the aquiris channels suses only segment 0

m_e_eV = 0.510998928e6 # http://physics.nist.gov/cgi-bin/cuu/Value?me 2014-04-21
c_0_mps = 299792458 # http://physics.nist.gov/cgi-bin/cuu/Value?c|search_for=universal_in! 2014-04-21

sourceDict = {}
def getSource(sourceString):
    global sourceDict
    if sourceString not in sourceDict:
        sourceDict[sourceString] = psana.Source(sourceString)
    return sourceDict[sourceString]

def timeScaleFromDataSource(ds, sourceString):
    '''Returnes time scale of aquiris trace from datasource object
    
    Returns None at failiure.
    Unit is microseconds.'''
    
    # Get the configuration
    try:
        acqirisConfig = ds.env().configStore().get(psana.Acqiris.ConfigV1,
                getSource(sourceString) )
    except:
        return None
        
    # make the time scale vector for the acqiris channel.
    # This is just for convenience
    timeScale = acqirisConfig.horiz()
    # Start time
    t0 = timeScale.delayTime()
    # Time step
    dt = timeScale.sampInterval()
    # Number of samples
    nSample = timeScale.nbrSamples()
    # Make the time scale vector from the above information and rescale it
    # to microseconds 
    return np.arange(t0, dt*nSample, dt)*1e6

def getTimeSlice(timeScale, tMin, tMax):
    return slice(
            timeScale.searchsorted(tMin),
            timeScale.searchsorted(tMax)
            )

_vertScaling = {}
_vertOffset = {}
def rescaleToVolts(rawTraces, channel, ds=None, sourceString=None):
    if type(channel) == list:
        returnList = True
    else:
        returnList = False
        channel = [channel]
        rawTraces = [rawTraces]
    
    if (ds is not None) and (sourceString is not None): 
        if any( [ ch not in _vertScaling for ch in channel ] ):
            setupVoltageRescalingFromDataSource(ds, sourceString, ch)

    
    rescaled = [tr * _vertScaling[ch] - _vertOffset[ch] 
            for tr, ch in zip(rawTraces, channel)]
    return rescaled if returnList else rescaled[0]
        

def setupVoltageRescalingFronDataSource(ds, sourceString, channel):
    global _vertScaling, _vertOffset 
    # Get the configuration
    try:
        acqirisConfig = ds.env().configStore().get(psana.Acqiris.ConfigV1,
                getSource(sourceString) )
    except:
        return None

    for ch in (channel if type(channel)==list else [channel]):
        # Get the scaling constants for the vertical scale.
        # convenience reference
        vertScale = acqirisConfig.vert()[ch]
        # The vertical scale information is given as the full scale voltage over
        # all the 2**16 bits.
        # Here the voltage per bit is calculated
        _vertScaling[ch] = vertScale.fullScale() * 2**-16
        # The scale also has an offset in voltage
        _vertOffset[ch] = vertScale.offset()

    
def timeTraceFromEvent(evt, sourceString, channel):
    try:
        # try to get the acqiris data
        acqirisData = evt.get(psana.Acqiris.DataDescV1, getSource(sourceString))
    except:
        return None
    
    wf = []
    if type(channel) == list:
        returnList = True
    else:
        returnList=False
        channel = [channel]

    for ch in channel:
        wf.append( acqirisData.data(ch).waveforms()[segment] )
    
    return wf if returnList else wf[0]

def getTimeToEnergyConversion(tScale_us, EScale_eV, D_mm, t0_us, E0_eV=0):
    # Bin edges of the given time axis
    dt = np.diff(tScale_us).mean()
    tEdges = np.concatenate([tScale_us - dt/2,
        tScale_us[-1:] + dt/2])
    ntBins = len(tScale_us)
    # and the corresponding bin edges in the energy domain. 
    # The conversion is given by:
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
    eEdges = D_mm**2 * m_e_eV * 1e6 / (c_0_mps**2 * 2 * (tEdges - t0_us)**2) + E0_eV

    #Bin edges of the energy scale
    dE = np.diff(EScale_eV).mean()
    EEdges = np.concatenate([EScale_eV - dE/2,
                EScale_eV[-1:] + dE/2])
    nEBins = len(EScale_eV)
    
    # Make matrixes out of the edges vectors in energy domain
    Me = np.concatenate([eEdges.reshape(1,-1)] * nEBins)
    ME = np.concatenate([EEdges.reshape(-1,1)] * ntBins, axis=1)

    # Compute the start and end energies for the conversion from the time axis
    # to energy axis
    highE = ( np.minimum( Me[:,:-1], ME[1:,:] ) )
    lowE = ( np.maximum( Me[:,1:], ME[:-1,:] ) )
    # Only where the high energy is more than the low energy the conversion makes any sense
    I = lowE < highE
    # Allocate a tempoaraty conversion matrix
    tempMat = np.zeros((nEBins, ntBins))
    # Calculate the elements of the conversion matrix
    # For each time bin the measured amplitude is multiplied by the bin size
    # in order to arrive at the integral of the signal. Then it is
    # determined how much of each time bin contributes to each energy bin.
    # This is done by comparing the edge positions of the time and energy
    # bins and assigning the correct proportion of the integral in time
    # domain to integral in the energy domain. Finally the total integral is
    # divided by the energy bin size in order to return to an amplitude.
    # Summation over all time bins is performed in the matrix multiplication
    # of the conversion matrix with the time domain amplitude vector.
    tempMat[I] = dt * (highE[I] - lowE[I]) / ( Me[:,:-1] - Me[:,1:] )[I] / dE
    # The conversion matrix is highly sparse, thus make a sparse matrix to
    # speed up the calculationss
    return coo_matrix(tempMat)


# Test the module
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    # Ger a psana datasource
    print 'Connecting to datasource'
    ds = psana.DataSource('exp=amoc8114:run=24')

    tofSourceString = 'DetInfo(AmoETOF.0:Acqiris.0)'
    channel = 1

    print 'Get time Scale'
    timeScale = timeScaleFromDataSource(ds, tofSourceString)

    print 'Get time slice'
    tSlice = getTimeSlice(timeScale, 1.4, 1.9)
    timeScale = timeScale[tSlice]

    print 'Set up scaling'
    setupVoltageRescalingFronDataSource(ds, tofSourceString, channel)

    print 'Get time amplitude'
    timeAmplitude = timeTraceFromEvent(ds.events().next(), tofSourceString,
            channel)[tSlice]

    print 'Rescaling to volts'
    trace_V = -rescaleToVolts(timeAmplitude, channel)
    trace_V -= trace_V[timeScale < 1.5].mean()
    
    plt.figure(1); plt.clf()
    plt.subplot(221)
    plt.plot(timeScale, timeAmplitude)
    plt.title('raw amplitude')

    plt.subplot(223)
    plt.plot(timeScale, trace_V)
    plt.title('Volt converted (and inverted)')

    tofCalib = {
            'D_mm' : 600,
            't0_us' : 1.51
            }
    energyScale = np.linspace(0, 200, 1000)

    print 'Get energy conversion matrix'
    tToE = getTimeToEnergyConversion(timeScale, energyScale, **tofCalib)
    eTrace = tToE.dot(trace_V)
    
    plt.subplot(222)
    plt.imshow(tToE.todense(), interpolation='none', aspect='auto')

    plt.subplot(224)
    plt.plot(energyScale, eTrace)
