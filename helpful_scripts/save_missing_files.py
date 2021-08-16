

import sys, os 

input_file = open('ModifiedDataset/missing_enhanced.txt', 'r')

lines = input_file.readlines()

for line in lines:
    print('filename', line)

    line = line[:-1]

    pid = line.split('_')[0]

    cp_command = 'cp ModifiedDataset/Input/' + pid + '/' + line + ' noteable_images/missing'

    print(cp_command)

    os.system(cp_command)