import subprocess
import sys

if len(sys.argv) < 2:
    print 'to few args'
    sys.exit()

startRun = int(sys.argv[1])

if len(sys.argv) < 3:
    endRun = startRun
else:
    endRun = int(sys.argv[2])
endRun += 1

for run in range(startRun, endRun):

    command = 'bsub -q psanaidleq -o logFiles/run{0}.log -J run{0}'.format(run)
    command += ' python mainAnalysis.py exp=amoc8114:run={0} run{0}_all.h5'.format(run)
    print command
    subprocess.call(command.split())
