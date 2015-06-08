import argparse

# A command line parser
def parse_cmd_line():
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

    parser.add_argument(
            '--overwrite',
            default = False,
            action = 'store_true',
            help = ('Use the settings given in the hdf5 file but overwrite'
                +' all the data in the file.'))
    parser.add_argument(
            '-u', '--update',
            action = 'append',
            type = str,
            metavar = 'dataName',
            default = [],
            help = '''Name of data in hdf5 file to update.''')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_cmd_line()
    print args
