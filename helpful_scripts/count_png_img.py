'''
count_png_img.py

Returns the number of png images in the given command line argument representing
the root of some directory structure 

'''


import os

def main(root_dir):
    num_png = 0

    for (root, dirs, files) in os.walk(root_dir, topdown=True):
        for filename in files:
            if '.png' in filename:
                num_png += 1

    print("There are", num_png, "png files in the directory structure with root", root_dir)


if __name__ == "__main__":
   main(sys.argv[1])

