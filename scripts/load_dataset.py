''' 
Loads dataset object from h5 and csv files.


'''

import csv
import h5py
import numpy as np


from PIL import Image

from os import listdir
from os.path import isfile, join


# My own helper scripts
from myconfig import *
from helper_functions import *



def read_patches_and_meta(image_id):
    ''' Reads num_patches patches and meta into numpy arrays. '''

    csv_name = "101527_seg_P128_L13_D8.csv"
    hdf_name = "101527_seg_P128_L13_D8.h5"
    
    patches_, coords_, labels_ = [], [], []
    
    # First read patch meta data
    with open(test_db_dir + csv_name, newline='') as metafile:
        reader = csv.reader(metafile, delimiter=' ', quotechar='|')
        for i, row in enumerate(reader):
            labels_.append(int(row[0]))
            coords_.append([int(row[1]), int(row[2])])

    # Now load the images from H5 file
    file = h5py.File(test_db_dir + hdf_name,'r+')
    dataset = file['/' + 't']
    new_patches = np.array(dataset).astype('uint8')
    for i, patch in enumerate(new_patches):
        patches_.append(patch)

    file.close()
    return np.array(patches_), np.array(labels_), np.array(coords_)



cases = load_cases(csv_name)
# Insert any k-split selection here

print("Selected cases read", cases)
print("Total cases used: " + str(len(cases)))


# a_patches, a_labels, a_coords = np.zeros((1, new_patch_dim, new_patch_dim, 3)), [], np.zeros((1, 2, 1))

start_time = start_timer()

for case in cases:
    print("================ Loading from Case", case, "================")
    # Get image IDs for each case
    case_file_dir = img_dir + folder_prefix + str(case) + "/"
    case_files = np.array([file for file in listdir(case_file_dir) if isfile(join(case_file_dir, file)) and '.svs' in file])
    print("Files in case", case, ":", case_files)

    image_ids = [f[:-4] for f in case_files]

    for image_id in image_ids:

    	del patches
    	del labels
    	del coords

		base_name = str(image_id) + "_seg_P" + str(patch_size) + "_L" + str(level) + "_D" + str(downsample_factor)
		csv_file = base_name + ".csv"
		h5_file = base_name + ".h5"

		patches, labels, coords = read_patches_and_meta(image_id)


	# TODO Add to all loaded data.
	# a_patches

print("Time to load all patches: ", end="")
end_timer(start_time)


# TODO store them in memory as a Dataset class.

