

import os, sys 

def main(folder):
    for root, subdir, files in os.walk(folder):
        for file in files:
            new_fname = '_'.join(file.split('_')[:-1]) + '.png'

            mv_command = 'mv ' + os.path.join(root, file) + ' ' + os.path.join(root, new_fname) 

            os.system(mv_command)



if __name__ == "__main__":
   main(sys.argv[1])