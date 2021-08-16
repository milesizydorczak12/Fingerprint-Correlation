# sd301a, sd302 checksums verified

import os, sys, subprocess, getopt

'''
Returns list of tuples (desired checksum, file name)
based on the checksum .csv file in this subdir
'''
def readChecksums(subdir, checksumFilename):
    checksumFilepath = os.path.join(subdir, checksumFilename)
    with open(checksumFilepath, 'r') as fin:
        print('Checking checksums according to checksum filepath:', checksumFilepath)
        fin_lines = fin.readlines()
        fin_lines = fin_lines[1:]
        fin_lines = [tuple([item.strip() for item in currLine.split(',')]) for currLine in fin_lines]
        #print(fin_lines[0])
        return fin_lines
    raise ValueError('Could not read checksums!')

'''
Returns checksum for file at this path
'''
def calcChecksum(filepath):
    raw_checksum_output = subprocess.check_output(['shasum', '-a', '256', filepath])
    checksum = raw_checksum_output.decode('utf-8').split()[0]
    #print(checksum)
    return checksum

'''
-> Checks checksums for just this directory.
Assumes that this is a directory with a given checksum.
-> Prints out the directory root name.
'''
def checkChecksums(subdir, checksumFilenameCsv):
    #print('Checking checksums for', subdir)
    for (desiredChecksum, filename) in readChecksums(subdir, checksumFilenameCsv):
        filepath = os.path.join(subdir, 'png', filename)
        if not os.path.exists(filepath):
            print(filepath + ' does not exist')
            return False
        calculatedChecksum = calcChecksum(filepath)
        if desiredChecksum != calculatedChecksum:
            print('Wrong checksum for ' + filename + '\nDesired:', desiredChecksum, '\nActual:', calculatedChecksum)
            return False
    return True


'''
-> Goes through ALL subdirectories of entire root directory
and checks checksums in ALL folders wherever checksums are given.
-> Makes sure that all files with known checksums exist.
'''
def checkAllChecksums(data_dir):
    for (root,dirs,files) in os.walk(data_dir, topdown=True):
        for filename in files:
            if all(keyword in filename for keyword in ('checksum', '.csv')):
                if not checkChecksums(root, filename):
                    return False
    return True

'''
Returns the root directory where all the fingerprint image data is stored
Usage: check_checksums.py --data_dir <data_dir>
'''
def parseArgs():
    argv = sys.argv[1:]
    data_dir = None
    usage_msg = "Usage: check_checksums.py --data_dir <data_dir>"
    try:
        opts, args = getopt.getopt(argv,"h?d:",["help", "data_dir="])
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(1)
    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print(usage_msg)
            sys.exit(0)
        elif opt in ('--data_dir', '-d'):
            data_dir = arg
    if data_dir is None:
        raise ValueError(usage_msg)
    if not os.path.exists(data_dir):
        raise ValueError(data_dir + ' does not exist')
    return data_dir

if __name__ == '__main__':
    data_dir = parseArgs()
    print('Good checksums:', checkAllChecksums(data_dir))
