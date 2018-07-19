
# coding: utf-8

# # Patch sampling from WSI images
# Author: @ysbecca
# 
# Using selected cases for exploratory stage.

# In[19]:

import csv
import h5py
import numpy as np
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator
from shapely.geometry import Polygon, Point
from scipy.misc import imresize
from PIL import Image

from os import listdir
from os.path import isfile, join

import json

# My own helper scripts
from myconfig import *
from helper_functions import *

# In[7]:


cases = load_cases(csv_name)

# For testing time requirements
# start_range = 0
# end_range = 2
# cases = cases[start_range:end_range]

total_cases = len(cases)


print("Selected cases read", cases)
print("Total cases used: " + str(total_cases))

# Set directories for svs files and json annotations.

# In[18]:



# In[9]:

# HELPER FUNCTIONS **********************************

def patch_to_tile_size(patch_size, overlap):
    return patch_size - overlap*2


def fetch_regions(image_id):
    path = json_dir + str(image_id)
    
    regions, region_labels = [], []
    try:
        with open(path + "_EP.json") as f_:
            data = json.load(f_)
        # Downsample regions before saving
        # print("Found ", len(data), "regions in EP file.")
        
        data_ = np.array([np.array(xi) for xi in data])
        for d in data_:
            region_labels.append("EP")
            regions.append(np.multiply(np.array(d), 1.0/annotation_downsample))
            
    except(FileNotFoundError, IOError):
        print("No EP file available for " + str(image_id))
    try:
        with open(path + "_SM.json") as f_:
            data = json.load(f_)
        # Downsample regions before saving
        # print("Found ", len(data), "regions in SM file.")        
        
        data_ = np.array([np.array(xi) for xi in data])
        for d in data_:
            region_labels.append("SM")
            regions.append(np.multiply(np.array(d), 1.0/annotation_downsample))
            
            
    except(FileNotFoundError, IOError):
        print("No SM file available for " + str(image_id))

    return regions, region_labels

def generate_label(polys, region_labels, point):
    # regions = array of vertices (all_coords)
    # point [x, y]
    
    # see if point is in any of regions provided
    for i in range(len(region_labels)): 
        if polys[i].contains(Point(point[0], point[1])):
            return label_map[region_labels[i]]
    else:
        return -1 # not in any of the regions - unknown label


def store_hdf5(patches, dir_path, coords, labels, csv_name):
    ''' Saves patches into hdf5 files, and meta into csv files '''

    print("Writing csv to:", csv_name + '.csv')

    # First write meta.
    with open(dir_path + csv_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(labels)):
             # Last entry is is_pseudo_label, set to 1 when labeling
            row = [labels[i], coords[i][0], coords[i][1], 0]
            writer.writerow(row)

    print("Wrote csv.")
    file = h5py.File(dir_path + csv_name + '.h5','w')
    print("Created file", file)
    dataset = file.create_dataset('t', np.shape(patches), h5py.h5t.STD_I32BE, data=patches, compression="gzip", compression_opts=9)

    print("===================")
    print("Meta data saved in", csv_name + ".csv")
    print("Dataset dataspace is " + str(dataset.shape))
    print("Dataset numpy datatype is " + str(dataset.dtype))
    print("Dataset name is " + str(dataset.name))
    print("Dataset was created in the file " + str(dataset.file))
    print("===================")
    file.close()

# END helper functions ************************************************

# In[10]:

tile_size = patch_to_tile_size(patch_size, pixel_overlap)

for case in cases:
    print("================ Sampling from Case", case, "================")
    # Get image IDs for each case
    case_file_dir = img_dir + folder_prefix + str(case) + "/"
    case_files = np.array([file for file in listdir(case_file_dir) if isfile(join(case_file_dir, file)) and '.svs' in file])
    print("Files in case", case, ":", case_files)

    image_ids = [f[:-4] for f in case_files]

    for image_id in image_ids:

        start_time = start_timer()


        slide = open_slide(case_file_dir + image_id + ".svs")
        tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=pixel_overlap, limit_bounds=True)


        # In[11]:

        tiles.level_tiles[level]

        # In[14]:

        x_tiles, y_tiles = tiles.level_tiles[level]

        print("x tiles:", x_tiles, "and y tiles:", y_tiles)

        x, y = 0, 0
        count = 0
        patches, coords, labels = [], [], []

        regions, region_labels = fetch_regions(image_id)
        polys = [Polygon(r) for r in regions]

        while y < y_tiles:
            while x < x_tiles:
                new_tile = np.array(tiles.get_tile(level, (x, y)), dtype=np.uint8)
                new_tile = imresize(new_tile, (new_patch_dim, new_patch_dim))
                
                # OpenSlide calculates overlap in such a way that sometimes depending on the dimensions, edge 
                # patches are smaller than the others. We will ignore such patches.
                if np.shape(new_tile) == (new_patch_dim, new_patch_dim, 3):
                    patches.append(new_tile)
                    coords.append(np.array([x, y]))
                    count += 1

                    # Calculate the patch label based on centre point.
                    point = tiles.get_tile_coordinates(level, (x, y))[0]
         
                    point_ = (point[0]/annotation_downsample, point[1]/annotation_downsample)
                    labels.append(generate_label(polys, region_labels, point_))
                    
                x += 1
            y += 1
            x = 0

        print("Patch sampling for image " + str(image_id) + " done: ", end="")
        end_timer(start_time)

        # In[17]:

        print(np.shape(patches))
        print(np.shape(coords))
        print(np.shape(labels))


        # Save patches and meta in hdf5/csv files.

        # In[24]:

        # In[25]:

        # Store images by image_id + seg + patch_size + level + downsample_factor
        csv_name = str(image_id) + "_seg_P" + str(patch_size) + "_L" + str(level) + "_D" + str(downsample_factor)
        store_hdf5(patches, test_db_dir, coords, labels, csv_name)


# In[26]:



print("Exiting now.")

