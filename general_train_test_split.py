# correctness has been manually verified (looked at file directory to make sure there is correct split and # of images)

import os
import shutil
import sys
import getopt
import statistics

from fileProcessingUtil import *

def get_pid_fgrp(filename):
    return (get_id(filename), get_fgrp(filename))

# Precondition:
#   [root_dir] is the directory storing all the gathered nist data;
#    its immediate subdirectories are train and val
#    data is organized by pid
def split(root_dir, val_fgrps, train_fgrps):
    # keep track of how many samples per person and fgrp
    pid_2_numSamples = {'val':dict(),'train':dict()}
    fgrp_2_numSamples = {'val':dict(),'train':dict()}
    pid_2_numUsedSamples = {'val':dict(),'train':dict()}
    fgrp_2_numUsedSamples = {'val':dict(),'train':dict()}

    # file directories
    data_path = root_dir

    val_path = os.path.join(data_path, 'val')
    train_path = os.path.join(data_path, 'train')
    #print(val_path, train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(train_path)

    unused_val_path = os.path.join(data_path, 'val_unused')
    unused_train_path = os.path.join(data_path, 'train_unused')
    os.makedirs(unused_val_path, exist_ok=True)
    os.makedirs(unused_train_path, exist_ok=True)

    # move everything to corresponding unused folders
    for group_path, unused_path in zip([val_path, train_path], [unused_val_path, unused_train_path]):
        for (root, dirs, files) in os.walk(group_path):
            for file in files:
                if '.png' in file:
                    pid = root.split('/')[-1]
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(unused_path, pid)
                    os.makedirs(dest_path, exist_ok=True)
                    if os.path.join(dest_path, file) != source_path:
                        shutil.move(source_path, dest_path)

    # count number of samples
    for group, unused_path in zip(['val', 'train'], [unused_val_path, unused_train_path]):
        for pid in os.listdir(unused_path):
            for img in os.listdir(os.path.join(unused_path, pid)):
                _, curr_fgrp = get_pid_fgrp(img)
                if pid not in pid_2_numSamples[group]:
                    pid_2_numSamples[group][pid] = 0
                pid_2_numSamples[group][pid] += 1
                if curr_fgrp not in fgrp_2_numSamples[group]:
                    fgrp_2_numSamples[group][curr_fgrp] = 0
                fgrp_2_numSamples[group][curr_fgrp] += 1

    # conduct the split: move fingerprints from unused folders to train & test folders
    for i in [(val_fgrps, val_path, unused_val_path, 'val'), (train_fgrps, train_path, unused_train_path, 'train')]:
        the_fgrps, new_path, unused_path, group = i[0], i[1], i[2], i[3]
        for fgrp in the_fgrps: # for all fingerprints in this train/test set
            for pid in os.listdir(unused_path): # currently, everything in unused
                for img in os.listdir(os.path.join(unused_path, pid)):
                    _, curr_fgrp = get_pid_fgrp(img)
                    if int(curr_fgrp) == int(fgrp):
                        source_path = os.path.join(unused_path, pid, img)
                        os.makedirs(os.path.join(new_path, pid), exist_ok = True)
                        dest_path = os.path.join(new_path, pid, img)
                        shutil.move(source_path, dest_path)

                        if pid not in pid_2_numUsedSamples[group]:
                            pid_2_numUsedSamples[group][pid] = 0
                        pid_2_numUsedSamples[group][pid] += 1
                        if fgrp not in fgrp_2_numUsedSamples[group]:
                            fgrp_2_numUsedSamples[group][fgrp] = 0
                        fgrp_2_numUsedSamples[group][fgrp] += 1

    for group in ['train', 'val']:
        print('For group', group)
        """
        print('\tnumber of samples per person id:')
        for the_id in sorted(pid_2_numSamples[group]):
            print('\t\t' + the_id + ':' , pid_2_numSamples[group][the_id])
        print('\tnumber of samples per fingerprint:')
        for the_fgrp in sorted(fgrp_2_numSamples[group]):
            print('\t\t' + the_fgrp + ':', fgrp_2_numSamples[group][the_fgrp])
        """
        print('\taverage number of samples per person id:',
        sum(pid_2_numSamples[group].values()) / len(pid_2_numSamples[group].values())
        )
        print('\ttotal number samples:', sum(pid_2_numSamples[group].values()))
        print('\ttotal number samples:', sum(fgrp_2_numSamples[group].values()))

        """
        print('\tnumber of used samples per person id:')
        for the_id in sorted(pid_2_numUsedSamples[group]):
            print('\t\t' + str(the_id) + ':' , pid_2_numUsedSamples[group][the_id])
        print('\tnumber of used samples per fingerprint:')
        for the_fgrp in sorted(fgrp_2_numUsedSamples[group]):
            print('\t\t' + str(the_fgrp) + ':', fgrp_2_numUsedSamples[group][the_fgrp])
        """
        print('\taverage number of used samples per person id:',
        sum(pid_2_numUsedSamples[group].values()) / len(pid_2_numUsedSamples[group].values())
        )
        print('\tstandard deviation of used samples per fingerprint:',
            statistics.stdev(pid_2_numUsedSamples[group].values()))
        print('\ttotal number used samples:', sum(pid_2_numUsedSamples[group].values()))
        print('\ttotal number used samples:', sum(fgrp_2_numUsedSamples[group].values()))

    return

def main(root_dir, val_fgrps, train_fgrps):
    print('validation fingerprints', val_fgrps)
    print('train fingerprints', train_fgrps)
    split(root_dir, val_fgrps, train_fgrps)
    return

def parseArgs():
    argv = sys.argv[1:]
    usage_msg = "Usage: general_train_test_split.py --root_dir <root_dir> --val_fgrps <val_fgrps> --train_fgrps <train_fgrps>"
    try:
        opts, args = getopt.getopt(argv,"h?r:v:t:",["help", "root_dir=", "val_fgrps=", "train_fgrps="])
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(1)

    root_dir = val_fgrps = train_fgrps = None

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print(usage_msg)
            sys.exit(0)
        elif opt in ('--root_dir', '-r', '--root'):
            root_dir = arg
        elif opt in ('--val_fgrps', '-v', '--val'):
            val_fgrps = [int(i) for i in arg.split()]
        elif opt in ('--train_fgrps', '-t', '--train'):
            train_fgrps = [int(i) for i in arg.split()]

    all_fgrps_used = set()
    all_fgrps_used.update(val_fgrps + train_fgrps)

    if len(all_fgrps_used) != len(val_fgrps + train_fgrps):
        raise ValueError('train and val must be mututally exclusive')

    if root_dir is None or val_fgrps is None or train_fgrps is None:
        raise ValueError(usage_msg)

    return root_dir, val_fgrps, train_fgrps

if __name__ == '__main__':
    root_dir, val_fgrps, train_fgrps = parseArgs()
    main(root_dir=root_dir, val_fgrps=val_fgrps, train_fgrps=train_fgrps)
