import psana
import numpy as np

segment = 0 # Somehow the aquiris channels suses only segment 0

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


# Test the module
if __name__ == '__main__':
    # Ger a psana datasource
    print 'Connecting to datasource'
    ds = psana.DataSource('exp=amoc8114:run=24')

    tofSourceString = 'DetInfo(AmoETOF.0:Acqiris.0)'
    print 'Get time Scale'
    timeScale = timeScaleFromDataSource(ds, tofSourceString)
