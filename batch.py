import subprocess
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('firstRun', type=int)
parser.add_argument('lastRun', type=int, nargs='?', default=None)
parser.add_argument('--overwrite', default=False, action='store_true')

args = parser.parse_args()
if args.lastRun is None:
    args.lastRun = args.firstRun


for run in range(args.firstRun, args.lastRun+1):
    hFileName = 'data/run{}_all.h5'.format(run)
    if not os.path.exists(hFileName):
        sourceNeeded = True 
    else:
        sourceNeeded = False

    with open('logFiles/run{}.log'.format(run), 'w') as fp:
        pass


    command = 'bsub -q psanaidleq -o logFiles/run{0}.log -J run{0}'.format(run)
    command += ' python buildH5.py -v'
    if sourceNeeded:
        command += ' exp=amoc8114:run={}'.format(run)
    command += ' ' + hFileName
    if args.overwrite:
        command += ' --overwrite'
    print command
    subprocess.call(command.split())
