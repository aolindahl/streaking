import psana
import numpy as np

def timeScaleFromDataSource(ds):
    '''Returnes time scale of aquiris trace from datasource object
    
    Returns None at failiure.
    Unit is microseconds.'''
    
    segment = 0 # Somehow the aquiris channels suses only segment 0

    # Get the configuration
    try:
        acqirisConfig = ds.env().configStore().get(psana.Acqiris.ConfigV1)
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

    

# Test the module
if __name__ == '__main__':
    # Ger a psana datasource
    print 'Connecting to datasource'
    ds = psana.DataSource('exp=amoc8114:run=24')

    print 'Get time Scale'
    timeScale = timeScaleFromDataSource(ds)
