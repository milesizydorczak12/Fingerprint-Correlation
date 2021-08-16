
import os, sys 


def main(folder):
    for root, subdir, files in os.walk(folder):
        for file in files:
            pid = file.split('_')[0]
            os.makedirs(pid, exist_ok = True)

            mv_command = 'mv ' + file + ' ' + pid + '/' 

            os.system(mv_command)


if __name__ == "__main__":
   main(sys.argv[1])