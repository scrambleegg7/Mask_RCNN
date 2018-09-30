#
import pandas as pd  
import shutil
import numpy as np  
import os

from glob import glob

def fileProcess():


    ROOT_DIR = "/Users/donchan/Documents/Miyuki/ssd_prescription/results"
    MOVE_DIR = "/Users/donchan/Documents/Miyuki/ssd_prescription/results/consolidate"

    dirs = os.listdir(ROOT_DIR)

    for l in dirs:
        print(l)
        img_files = [r for r in glob( os.path.join(ROOT_DIR,l,"*.jpg"))  ]

        
        for f in img_files:

            new_file = os.path.basename(f)
            new_file = os.path.join( MOVE_DIR, new_file )
            shutil.move(f,new_file)
        

def main():

    fileProcess()


if __name__ == "__main__":
    main()