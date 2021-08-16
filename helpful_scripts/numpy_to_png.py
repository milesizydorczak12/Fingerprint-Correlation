'''
numpy_to_png.py

Convert all numpy array files in the given file structure to png images

'''


import sys, os 
from matplotlib import pyplot as plt
import numpy as np

def main(source_path, dest_path):
    for root, dirs, files in os.walk(source_path):
        pid = os.path.basename(root)
        if not pid:
            continue
        result_person_dir = os.path.join(dest_path, pid)
        os.makedirs(result_person_dir, exist_ok = True)
        for file in files:
            result_matrix = np.load(os.path.join(root, file))
            png_file = file.split('.')[0] + '.png'
            plt.axes([0,0,1,1])
            plt.axis("off")
            plt.imsave(os.path.join(result_person_dir, png_file), result_matrix, cmap='coolwarm')


if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2])



