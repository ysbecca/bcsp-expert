
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
from xml.dom import minidom

from os import listdir
from os.path import isfile, join

import json
import sys
# My own helper scripts
from myconfig import *
from helper_functions import *

# In[7]:


cases, gtruth = load_cases(csv_name, is_test=True)

cases = cases[case_start:case_end]
gtruth = gtruth[case_start:case_end]

# hello....

# For testing time requirements
# start_range = 0
# end_range = 2
# cases = cases[start_range:end_range]

total_cases = len(cases)
print(gtruth)

print("Selected cases read", cases)
print("Total cases used: " + str(total_cases))

# Set directories for svs files and json annotations.

# In[18]:



# In[9]:

# HELPER FUNCTIONS **********************************

def load_xml_regions(image_id):
    ''' Load XML regions. '''

    path = xml_dir + str(image_id) + ".xml"
    errors = []
    print(path)
    error = 0

    regions = []
    try:
        xml = minidom.parse(path)
    except:
        errors.append(path)
        error = 1
        print("ERROR reading xml file..........................")
        
    if not error:
        regions_ = xml.getElementsByTagName("Region")
        region_ = []

        for r in regions_:
            vertices = r.getElementsByTagName("Vertex")
            coords = np.zeros((len(vertices), 2))

            for i, vertex in enumerate(vertices):
                coords[i][0] = vertex.attributes['X'].value
                coords[i][1] = vertex.attributes['Y'].value
            region_.append(coords)
        regions.append(np.multiply(np.array(region_), 1.0/annotation_downsample))

    return regions

def calculate_bounding_points(regions):
    max_points = []
    min_points = []
    for r in regions:
        max_x, max_y = np.amax(r, axis=0)
        min_x, min_y = np.amin(r, axis=0)
        max_points.append([max_x, max_y])
        min_points.append([min_x, min_y])

    return max_points, min_points

def check_is_roi(polys, regions, max_pts, min_pts, point):
    # regions = array of vertices (all_coords)
    # point [x, y]

    # Optimisation of checking -- first check coordinates are in any bounding box of the polys.
    # If not then don't bother checking polygon.
    within_boundaries = False
    for i in range(len(regions)):
        if point[0] >= min_pts[i][0] and point[0] <= max_pts[i][0] and point[1] >= min_pts[i][1] and point[1] <= max_pts[i][1]:
            if polys[i].contains(Point(point)):
                return 1
    return 0 # Not in any ROI

def image_is_background(image):
    '''
    Determines using a simple threshold whether the patch is primarily background.
    '''
    threshold = 0.25
    x, y, dim = np.shape(image)
    if (np.count_nonzero(np.where(image > 210, 0, image))/(x*y*dim) < threshold):
        return True
    else:
        return False

def store_csv_meta(dir_path, coords, labels, rois, csv_name):
    print("Writing csv to:", csv_name + '.csv')

    # First write meta.
    with open(dir_path + csv_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(labels)):
            row = [labels[i], coords[i][0], coords[i][1], rois[i]]
            writer.writerow(row)

    print("Wrote csv.")

def store_hdf5(datasets, patches, written_count):
    ''' Saves sub-set of patches into hdf5 file. '''
    p_count = len(patches[0])
    if p_count > 0:
        # Resize dataset, then write to new expanded shape.
        for i in range(samples_per_patch):
            datasets[i].resize((written_count + p_count), axis=0)
            datasets[i][written_count:(written_count + p_count)] = np.array(patches[i], dtype=np.int32)
        
            print("Dataset", i, "dataspace is " + str(datasets[i].shape))

    return datasets, written_count + p_count

# END helper functions ************************************************

# In[10]:

for c, case in enumerate(cases):
    print("================ Sampling from Case", case, "================")
    # Get image IDs for each case
    case_file_dir = img_dir + folder_prefix + str(case) + "/"
    case_files = np.array([file for file in listdir(case_file_dir) if isfile(join(case_file_dir, file)) and '.svs' in file])
    print("Files in case", case, ":", case_files)

    image_ids = [f[:-4] for f in case_files]

    for image_id in image_ids:
        csv_name = str(image_id) + "_T"

        # H5 file storage preparation.
        h5_files = []
        for j in range(samples_per_patch):
            h5_files.append(h5py.File(test_db_dir + csv_name + '_' + str(j) + '.h5','w'))

        print("Created files:", h5_files)
        datasets = []
        chunk_size = 1000
        for j in range(samples_per_patch):
            datasets.append(h5_files[j].create_dataset(
                'dataset',
                (0, base_patch_size, base_patch_size, 3), # shape
                h5py.h5t.STD_I32BE,
                maxshape=(None, base_patch_size, base_patch_size, 3),
                chunks=(chunk_size, base_patch_size, base_patch_size, 3)))
                # compression="gzip", 
                # compression_opts=9))

        start_time = start_timer()

        slide = open_slide(case_file_dir + image_id + ".svs")
        slide_dims = slide.level_dimensions[0]
        print("Slide dimensions:", slide_dims)

        initial_offset = int(patch_sizes[len(patch_sizes) - 1] / 2.0)

        x, y = initial_offset, initial_offset
        written_count, total_count = 0, 0
        patches, coords, labels, rois = [], [], [], []
        for i in range(samples_per_patch):
            patches.append([]) # samples_per_patch patches for each x, y sampling position

        base_patches = patches

        regions = load_xml_regions(image_id)[0]
        print("Found", len(regions), "region(s).")

        max_pts, min_pts = calculate_bounding_points(regions)

        polys = []
        for r in regions:
            try:
                polys.append(Polygon(r))
            except:
                pass
        print("Found", len(polys), "poly(s).")

        if gtruth[c] == 'EM':
            label = [1, 0]
        else:
            label = [0, 1]

        patches_per_batch = chunk_size * samples_per_patch
        batch_count = 0
        # For test only -- end_stop
        end_stop, stop = 10*samples_per_patch, False
        while y < slide_dims[1] + initial_offset:
            while x < slide_dims[0] + initial_offset:
                #print("x, y:", x, "/", slide_dims[0], ",", y, "/", slide_dims[1])
                is_roi = 0
                break_out = False
                for i in range(samples_per_patch):
                    if not break_out:

                        x_ = x - int(0.5*patch_sizes[i])
                        y_ = y - int(0.5*patch_sizes[i])
                        new_tile = np.array(slide.read_region((x_, y_), level, (patch_sizes[i], patch_sizes[i])))[:,:,:3]

                        # The background test and label is always based on the first patch size.
                        if i == 0: 
                            if not image_is_background(new_tile):
                                # Get roi label.
                                point_ = (x/annotation_downsample, y/annotation_downsample)
                                is_roi = check_is_roi(polys, regions, max_pts, min_pts, point_)

                                # Add label, coordinate, coords values only once.
                                labels.append(label)
                                rois.append(is_roi)
                                coords.append(np.array([x, y]))
                            else:
                                break_out = True # Out of the for loop; no saving.
                        
                        if not break_out:
                            # All patches downsampled to the base patch size.
                            new_tile = np.array(Image.fromarray(new_tile).resize((base_patch_size, base_patch_size)))                        
                            patches[i].append(new_tile)
                            total_count += 1
                            batch_count += 1
                            if batch_count >= end_stop:
                                stop = True
                
                if batch_count >= patches_per_batch:
                    # Write entire batch to h5 file and clear memory.
                    print("batch_count >= patches_per_batch at", batch_count, ", total_count", total_count)
                    datasets, written_count = store_hdf5(datasets, patches, written_count)
                    patches = base_patches # Reset.
                    batch_count = 0
                    if stop:
                        break

                x += patch_sizes[0] # Full patch stride.
                if stop:
                    break
            y += patch_sizes[0]
            x = initial_offset
 
        # Last batch (may be incomplete).
        if total_count > written_count:
            datasets, written_count = store_hdf5(datasets, patches, written_count)
            del patches

        print("Total count:    ", total_count)
        print("Written count:  ", written_count)

        print("Patch sampling for image " + str(image_id) + " done: ", end="")
        end_timer(start_time)

        # In[17]:
        print("Patch meta: ================================================= **")
        print(np.shape(rois))
        print(np.shape(coords))
        print(np.shape(labels))

        # Store images by image_id + _T for test set meta.
        store_csv_meta(test_db_dir, coords, labels, rois, csv_name)
        print("Meta data saved in", csv_name + ".csv")

        for k in range(samples_per_patch):
            h5_files[k].close()
        
# In[26]:

print("Exiting now.")

