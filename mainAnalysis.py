import h5py
import argparse
import numpy as np
import tof

psana = None

# Analysis configuration
tofSourceString = 'DetInfo(AmoETOF.0:Acqiris.0)'
timeSlice_us = [1.5, 1.7]


# A command line parser
def parseCmdline():
    "Function used to parse the commahd line."
    parser = argparse.ArgumentParser(
            description=('Tool to get data from xtc files into a custom hdf5 '
                + 'format.')
            )

    parser.add_argument(
            'dataSource',
            type = str,
            nargs = '?',
            default = None,
            help=('xtc-file or other description of the data that cen be used'
                + ' by psana. Example "exp=amoc8114:run=108". '
                + ':idx will be added as needed.' 
                + '\nThis could also be an hdf5 file previously created.')
            )

    parser.add_argument(
            'hdf5File',
            type=str,
            default = None,
            help=('Path to new or existing hdf5 file.')
            )

    parser.add_argument(
            '-n',
            '--numEvents',
            metavar='N',
            default = -1,
            type = int,
            help=('Number of events to process. The events will be distributed'
                + 'over the whole file')
            )

    parser.add_argument(
            '-v',
            '--verbose',
            action='store_true',
            default=False,
            help=('Print stuff to the terminal')
            )

    return parser.parse_args()


def connectToDataSource(dataSource, verbose=False):
    global psana
    import psana
    if ':idx' not in dataSource:
        dataSource += ':idx'
    ds = psana.DataSource(dataSource)
    if verbose:
        print 'Connected to data source {}.'.format(dataSource)

    run = ds.runs().next()
    return ds, run

 
def setupFiles(dataSource, hdf5FileName, numEvents=-1, verbose=False):
    # If a data source is given
    if dataSource is not None:
        if verbose:
            print 'Data source given.'
        # Simply connect
        ds, run = connectToDataSource(dataSource, verbose)

        # and make a new hdf5 file. Overwriting any old file.
        hFile = h5py.File(hdf5FileName, 'w')
        if verbose:
            print 'HDF5 file {} created wor writing.'.format(hFile.filename)

        # Add the data source information to the hdf5 file
        if verbose:
            print 'Add data source to HDF5 file.'
        hFile.attrs.create('dataSource', dataSource)

        # Add information about the number of events to be processed
        N = len(run.times())
        if numEvents > 0:
            N = min(N, numEvents)
        hFile.attrs.create('numEvents', N)

        # Add the start time infomration to the hdf5 rile
        t0 = run.times()[0]
        hFile.attrs.create('startTime_s', float(t0.seconds()))

        #hFile.create_group('eventData')
        #hFile.create_group('envData')

    else:
        # Open the hdf5 file for reading and writing.
        hFile = h5py.File(args.hdf5File, 'r+')
        if verbose:
            print 'HDF5 file {} opened for read and write.'.format(hFile.filename)

        # Check if any psana data are missing
        if verbose:
            print 'Checking for psana data setds in the hdf5 file.'

        # Flag for missing data sets
        missing = False
        # Get all the datasets that should be there
        sets = psanaEnvDataDefinition()
        sets.update( psanaEventDataDefinition() )
        # Go through all the datasets
        for set in sets:
            # snd see of they are there
            if set not in hFile.keys():
                if verbose:
                    print '"{}" is NOT there: PSANA needed.'.format(set)
                missing = True
                break
            else:
                if verbose:
                    print '"{}" is there.'.format(set)

        if missing:
            # Get the data source from the hdf5 file
            if verbose:
                print 'Get data source from hdf5 file.'
            args.dataSource = hFile.attrs.get('dataSource')
            # and connect
            ds, run = connectToDataSource(args.dataSource, verbose)
        else:
            de, run = None, None


    return ds, run, hFile

def psanaEnvDataDefinition(traceLength=None):
    # Specify environment data sets
    dataSets = {
            'timeScale_us' : {'shape' : (traceLength, ), 'dtype' : 'f'}
            }
    return dataSets


def psanaEventDataDefinition(numEvents=None):
    # Specify the datasets that should be avaliable for psana data
    dataSets = {
            'fiducial' : {'shape' : (numEvents,), 'dtype' : 'i'},
            'eventTime': {'shape' : (numEvents,), 'dtype' : 'f'}
            }
    return dataSets

def makeEventDatasets(hFile, dataSets):
    emptyDatasets = []
    for set, spec in dataSets.iteritems():
        if set not in hFile:
            emptyDatasets.append(set)
            hFile.create_dataset(set, **spec)
    return emptyDatasets

def getEnvData(hFile, ds, setNames):
    # Make sure the set names is in a list
    setNames = list(setNames)

    timeScale = 'timeScale_us'
    if (timeScale in setNames) and (timeScale not in hFile.keys()):
        # Get the time scale from the data
        fullTimeScale_us = tof.timeScaleFromDataSource(ds, tofSourceString)
        timeSlice = slice( fullTimeScale_us.searchsorted( np.min( timeSlice_us ) ),
                fullTimeScale_us.searchsorted( np.max( timeSlice_us ) )
                )
        hFile.create_dataset(timeScale, data = fullTimeScale_us[timeSlice])
        hFile.create_dataset('timeSlice',
                data = np.array( [timeSlice.start, timeSlice.stop] ) )



def getEventData(hFile, evt, setName, i, t_runStart):

    if 'fiducial' in list(setName):
        hFile['fiducial'][i] = evt.get(psana.EventId).fiducials()

    if 'eventTime' in list(setName):
        time = evt.get(psana.EventId).time()
        hFile['eventTime'][i] = time[0] + time[1]*1e-9 - t_runStart

#Running the snalysis
if __name__ == '__main__':
    #Parset the command line arguments
    args = parseCmdline()
    # Make a verbose flag
    verbose = args.verbose

    # If in verbose mode, print some information
    if verbose:
        print 'Argumets are:'
        print repr(args)

    ds, run, hFile = setupFiles(args.dataSource, args.hdf5File, args.numEvents,
            verbose)

    # if the psana data is connected some data is missing.
    # Go through the environement data
    if ds is not None:
        getEnvData(hFile, ds, psanaEnvDataDefinition())


    # Get basic information from the hdf5 file
    # Events to process
    N = hFile.attrs.get('numEvents')
    # Start time
    t0 = hFile.attrs.get('startTime_s')

    # Get the time scale of the tof trace
    timeScale_us = hFile['timeScale_us']
    timeSlice = slice( *hFile['timeSlice'] )

    # Make space for the events in the hdf5 file and get a list of empty data
    # sets
    psanaEventDataSets = psanaEventDataDefinition(N)
    emptyEventDatasets = makeEventDatasets(hFile, psanaEventDataSets)

    # Get a list of the time objects to use
    times = run.times()[:N]
    # go through the corresponding events
    for i, time in enumerate(times):
        evt = run.event(time)
        getEventData(hFile, evt, psanaEventDataSets.keys(), i, t0)
    
    # Close the hdf5 file
    hFile.close()
    if verbose:
        print 'HDF5 file closed.'
