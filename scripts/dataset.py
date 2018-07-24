'''

Dataset class (adapted using class from by Hvass-Labs tutorials) customised for whole slide image patches, 
and using py-wsi to load patches from LMDB.

Optionally performs basic augmentation of patches (8 total: rotations of 90 degrees * k, for k={1, 2, 3}
and flips across the horizontal and vertical axes.)

Author: Hvass-Labs and @ysbecca

'''
import math
import numpy as np
import random

from os import listdir
from os.path import isfile, join
import csv
import h5py

# My own helper scripts
from scripts.myconfig import *
from scripts.helper_functions import *


class DataSet(object):

  def __init__(self, images, coords, labels, image_ids):

    self._num_images = np.array(images).shape[0]
    self._images = images

    # Boolean array versions of ID
    self._labels = labels 
    # The source image labels
    self._image_ids = image_ids
    # Integer labels
    # self._ids = ids 
    self._coords = coords

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def image_ids(self):
    return self._image_ids

  @property
  def labels(self):
    return self._labels

  @property
  def coords(self):
    return self._coords

  @property
  def num_images(self):
    return self._num_images


  @property
  def epochs_completed(self):
    return self._epochs_completed

  def set_images(self, images):
    self._images = images

  def set_labels(self, labels):
    self._labels = labels

  def set_coords(self, coords):
  	self._coords = coords

  # Shuffles all patches in the dataset object.
  def shuffle_all(self):
    if self.num_images <= 1:
        print("Cannot shuffle when", self.num_images, "images in set.")
        return

    list_all = list(zip(self._images, self._labels, self._image_ids, self._coords))
    random.shuffle(list_all)
    self._images, self._labels, self._image_ids, self._coords = zip(*list_all)
    self._images = np.array(self._images)
    self._labels = np.array(self._labels)
    self._image_ids = np.array(self._image_ids)
    self._coords = np.array(self._coords)


  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_images:
      # Finished epoch
      self._epochs_completed += 1

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_images
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]

# Helper function which shuffles the object.
def shuffle_multiple(list_of_lists):
  new_list = []
  if(len(list_of_lists) == 0):
    print("ERROR: no elements in list of lists for shuffling.")
    return 0
  perm = np.arange(len(list_of_lists[0]))
  np.random.shuffle(perm)
  for list_ in list_of_lists:
    new_list.append(np.copy(list_[perm]))

  return new_list


def read_datasets(valid_id, train_id, k, shuffle_all=False, new_valid=True):
  class DataSets(object):
      pass
  dataset = DataSets()

  cases = load_cases(csv_name)
  print("Cases:", cases)

  # Divided by CASE, so no patient's images will be in both valid and train sets.
  dataset.valid = fetch_seg_dataset(valid_id, k, cases)
  dataset.train = fetch_seg_dataset(train_id, k, cases)

  if shuffle_all:
      dataset.train.shuffle_all()
      dataset.valid.shuffle_all()

  return dataset


def fetch_seg_dataset(k_set_id, k, cases, do_augments=False):

  selected = np.zeros((len(cases)))
  selected[k_set_id::k] = 1
  print("Selected:", selected)

  patches, coords, labels, image_ids = [], [], [], []

  for i, case in enumerate(cases):
    if selected[i]:
      case_file_dir = img_dir + folder_prefix + str(case) + "/"
      images = np.array([file for file in listdir(case_file_dir) if isfile(join(case_file_dir, file)) and '.svs' in file])

      # images[j][:-4] image ID
      for im in images:
        patches, coords, labels, image_ids = read_patches_and_meta_L(im[:-4], patches, coords, labels, image_ids)

  patches = np.array(patches)
  coords = np.array(coords)
  labels = np.array(labels)
  image_ids = np.array(image_ids)

  print("Finished loading k-set:", k_set_id)
  print(np.shape(patches))
  print(np.shape(coords))
  print(np.shape(labels))
  print(np.shape(image_ids))

  return DataSet(patches, coords, labels, image_ids)


def read_patches_and_meta_L(image_id, patches, coords, labels, image_ids):
    ''' Reads num_patches patches and meta of LABELLED patches into numpy arrays. '''

    file_name = str(image_id) + "_seg_P" + str(patch_size) + "_L" + str(level) + "_D" + str(downsample_factor) + "_L"
    
    try:
      # First read patch meta data
      with open(test_db_dir + file_name + ".csv", newline='') as metafile:
          reader = csv.reader(metafile, delimiter=' ', quotechar='|')
          for i, row in enumerate(reader):
              new_label = [0, 0]
              new_label[int(row[0])] = 1
              labels.append(np.array(new_label, dtype=np.int8))
              coords.append(np.array([int(row[1]), int(row[2])]))
              image_ids.append(image_id)

      # Now load the images from H5 file
      file = h5py.File(test_db_dir + file_name + ".h5",'r+')
      dataset = file['/' + 't']
      new_patches = np.array(dataset).astype('uint8')
      for i, patch in enumerate(new_patches):
          patches.append(patch)

      file.close()
    except:
      # No labelled patches exist for that case.
      pass
    return patches, coords, labels, image_ids

def stain_augment(patches):

  # Calculate max H, E, D range in dataset - with no colour normalisation.
  max_hed = [-2.0]*3
  min_hed = [1.0]*3

  for p in hed_patches:
    hed = np.dsplit(p, 3)
    for channel in range(3):
        max_hed[channel] = max(np.max(hed[channel]), max_hed[channel])
        min_hed[channel] = min(np.min(hed[channel]), min_hed[channel])

  print("Calculated max HED range:", max_hed)
  print("Calculated min HED range:", min_hed)

  # Tweak the H, E, and DAB channels within the range of the dataset (one channel at a time).
  tweaked_patches = []

  for p in hed_patches:
      channels = np.dsplit(p, 3)
      # Subtract and add by interval until hit min and max in any one pixel value
      interval = 0.03
      current = interval
      not_reached_max, not_reached_min = [True]*3, [True]*3
      while(True in (not_reached_max + not_reached_min)):
          for ch in range(3):
              if not_reached_max[ch]:
                  ch_copy = np.copy(channels)
                  new_channel = np.add(ch_copy[ch], current)
                  if np.count_nonzero(np.where(new_channel > max_hed[ch], 1, 0)) == 0:
                      ch_copy[ch] = new_channel
                      tweaked_patches.append(np.dstack(ch_copy))
                  else:
                      not_reached_max[ch] = False
              if not_reached_min[ch]:
                  ch_copy = np.copy(channels)
                  new_channel = np.subtract(ch_copy[ch], current)
                  if np.count_nonzero(np.where(new_channel < min_hed[ch], 1, 0)) == 0:
                      ch_copy[ch] = new_channel
                      tweaked_patches.append(np.dstack(ch_copy))
                  else:
                      not_reached_min[ch] = False
          current += interval

  print("Augmented patches via colour deconvolution:", len(tweaked_patches), "patches...")

  # return np.array()



def rotational_augment_patches(patches):
    # And mirroring too.
    aug_patches = []
    for im in patches:
        aug_patches.append(im) # Original
        aug_patches.append(np.fliplr(im))
        aug_patches.append(np.flipud(im))
        for rot in range(1, 4):
            aug_patches.append(np.rot90(im, rot))
            aug_patches.append(np.rot90(np.fliplr(im), rot))
    return np.array(aug_patches)









