
import sys, os 


def main(filepath):

    totals = {}

    for root, subdir, files in os.walk(filepath):
        for file in files:
            pid = file.split('_')[0]
            fingerID = (file.split('_')[-1]).split('.')[0]

            key = pid + fingerID

            if key in totals:
                totals[key] += 1
            else:
                totals[key] = 1

    print("min:", min(totals.items(), key=lambda x: x[1]))
    print("max:", max(totals.items(), key=lambda x: x[1]))



if __name__ == "__main__":
    main(sys.argv[1])







