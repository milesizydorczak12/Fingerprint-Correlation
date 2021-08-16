'''
split_files_by_sensor_letter.py

run with a command line arguments representing a source directory, a dest
dircetory, and a list of letters for the scanners that will be filtered into 
the destination.

'''

import os, sys, getopt
from shutil import copyfile

def main(argv):
    usage_msg = "Usage: python3 split_files_by_sensor_letter.py --src <src_dir> --dest <dest_dir> -l [l1, l2, ln]"

    SRC_DIR = ''
    DEST_DIR = ''
    sensor_letters = []


    try:
        opts, args = getopt.getopt(argv,"h?l:",["help", "letter=", "src=", "dest="])
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print(usage_msg)
            sys.exit(0)
        elif opt == '--src':
            SRC_DIR = arg
        elif opt in ('--letter', '-l'):
            sensor_letters.extend(['_' + l + '_' for l in arg.upper().split()])
        elif opt in ('--dest', '-d'):
            DEST_DIR = arg

    # Garuntee that user has provided src
    if not SRC_DIR or not DEST_DIR:
        print(usage_msg)
        sys.exit(1)

    for root, dirs, files in os.walk(SRC_DIR):
        pid = os.path.basename(root)
        if not pid:
            continue
        result_person_dir = os.path.join(DEST_DIR, pid)
        os.makedirs(result_person_dir, exist_ok = True)
        for file in files:
            for letter in sensor_letters:
                if letter in file:
                    copyfile(os.path.join(root, file), os.path.join(result_person_dir, file))


if __name__ == "__main__":
   main(sys.argv[1:])


