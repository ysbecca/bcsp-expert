import csv
import numpy as np
import sys

sys.path.append("../")

from py_wsi import *
import time
from datetime import timedelta

# First, load all the cases and associated labels from the csv.

start_time = time.time()

csv_name = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_cases.csv"
cases = []
with open(csv_name, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in r:
        cases.append(row[0])

# For testing limitations
# start_range = 0
# end_range = 2
# cases = cases[start_range:end_range]

total_cases = len(cases)

print("Selected cases read", cases)
print("Total cases: " + str(total_cases))


# Now use py_wsi to read in the WSI and sample patches.

# home_dir = "/nobackup/sc16rsmy/bcsp-expert-cases/"
# folder_prefix = "Case_"
# db_location = "/nobackup/sc16rsmy/bcsp-lmdb/"
# db_name = "training_set_annotated"

home_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp/"
folder_prefix = "Case_"
annotation_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_annotations/"
db_location = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_db/"
db_name = "training_set_annotated"

patch_size = 128
level = 15
overlap = 32 #in PIXELS

# label_map = {'SM': 0, 'EP': 1}

label_map = {'Normal': 0,
             'Benign': 1,
             'Carcinoma in situ': 2,
             'In situ carcinoma': 2,
             'Carcinoma invasive': 3,
             'Invasive carcinoma': 3,
            }

manager = Turtle(home_dir + folder_prefix, 
				db_location, 
				db_name, 
				xml_dir=annotation_dir, 
				label_map=label_map, 
				cases=cases,
				storage="hdf5")


print("Total WSI images:    " + str(manager.num_files))
print("HDF5 folder name:    " + str(manager.db_name))
print("File names:          " + str(manager.files))

# Now read the samples and store to LMDB.
manager.sample_and_store_patches(patch_size, level, overlap, load_xml=True)


print("Done sampling from all cases and saving to HDF5 files.")

end_time = time.time()
time_elapsed = str(timedelta(seconds=int(round(end_time - start_time))))
print("Time usage: " + time_elapsed)





