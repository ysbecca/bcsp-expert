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
from skimage.color import rgb2hed
from skimage.color import hed2rgb

# My own helper scripts
from myconfig import *
from helper_functions import *


class DataSet(object):

  def __init__(self, wsi_ids, rois=[]):

    # one k-set parameter
    self.wsi_ids = wsi_ids
    self.wsi_counts = [0] * len(wsi_ids)
    self.wsi_index = 0
    self._epochs_completed = 0
    self._index_in_epoch = 0


    # Boolean to indicate how mnay times we have iterated through all of the k-set.
    self.epoch_count = 0  

    # Helper variable for weird edge case in next_batch function.
    self.load_next_wsi = False

    # Set all parameters by loading the first set.
    self.read_next_patches_and_meta_test()

  def reset_all(self):
    ''' Resets the dataset to the first WSI of k-set. Does NOT reset epoch count. ''' 
    old_wsi_index = self.wsi_index

    self.wsi_index = 0
    self._index_in_epoch = 0
  
    # If next pointer was not at 1, then loaded set is NOT 0.
    if old_wsi_index != 1:
      self.read_next_patches_and_meta_test()

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
  def rois(self):
    return self._rois

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def reset_epoch_count(self):
    self.epoch_count = 0

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

    if len(self._rois) == 0:

      list_all = list(zip(self._images, self._labels, self._image_ids, self._coords))
      random.shuffle(list_all)
      self._images, self._labels, self._image_ids, self._coords = zip(*list_all)
      self._images = np.array(self._images)
      self._labels = np.array(self._labels)
      self._image_ids = np.array(self._image_ids)
      self._coords = np.array(self._coords)
    
    else:
      list_all = list(zip(self._images, self._labels, self._image_ids, self._coords, self._rois))
      random.shuffle(list_all)
      self._images, self._labels, self._image_ids, self._coords, self._rois = zip(*list_all)
      self._images = np.array(self._images)
      self._labels = np.array(self._labels)
      self._image_ids = np.array(self._image_ids)
      self._coords = np.array(self._coords)
      self._rois = np.array(self._rois)


  def next_batch(self, batch_size, get_roi=False, augment=False, get_coords=False, stop_at_epoch=False):
    """Return the next `batch_size` examples from this data set."""

    if self.load_next_wsi:
      self.read_next_patches_and_meta_test()
      self.load_next_wsi = False


    start = self._index_in_epoch

    print("wsi_index pointer -------------- ", self.wsi_index)
    
    # Check if we are looking at the last batch of k-set, smaller than the last image (Getting test/valid accuracy) and returning it.
    not_full_last_batch = (self._index_in_epoch + batch_size > self._num_images)
    if stop_at_epoch and not_full_last_batch:
      print("Returning uneven set - ", self._index_in_epoch, ":", self._num_images)

      # If this was the last batch of the last image, then update epochs.
      if self.wsi_index == 0:
        self.epoch_count += 1
        print("Setting epoch count to:", self.epoch_count)

      else:
        # Last batch of a non-last image.
        self.load_next_wsi = True

      # Reset pointer to first index again.
      self._index_in_epoch = 0


      if get_roi and not get_coords:
        return self._images[start:], self._labels[start:], self._rois[start:]
      elif get_roi and get_coords:
        return self._images[start:], self._labels[start:], self._rois[start:], self._coords[start:]
      else:
        return self._images[start:], self._labels[start:]

    elif not_full_last_batch:
      # If next set is last, then update epochs.
      if self.wsi_index == 0:
        self.epoch_count += 1
        print("Updated self.epoch_count to (last INCOMPLETE batch):", self.epoch_count)

      # We don't want the last incomplete batch so load next set.
      self._epochs_completed += 1

      # After resetting variables, fetch next wsi.
      self.read_next_patches_and_meta_test()

      return self.next_batch(batch_size, get_roi=get_roi, augment=augment, get_coords=get_coords, stop_at_epoch=stop_at_epoch)

    else: 

      # OTHERWISE, we have a FULL batch to return.
      self._index_in_epoch += batch_size
      end = self._index_in_epoch

      # If it's the last FULL batch of the LAST wsi, update epoch count and reset index pointer.
      future_batch_size_check = (self._num_images - end < batch_size) and not stop_at_epoch
      if self.wsi_index == 0 and (end == self._num_images or future_batch_size_check):

        self.epoch_count += 1
        print("Updated self.epoch_count to (last complete batch):", self.epoch_count)
        self._index_in_epoch = 0
        self._epochs_completed += 1

      # It's the last FULL batch of a NON-last wsi.
      elif (self._num_images - end == batch_size) and self.wsi_index > 0:
        self.load_next_wsi = True
        self._index_in_epoch = 0
        print("Last full batch of NON-last wsi.")

      print("Returning from", start, ":", end)

      if get_roi and not get_coords:
        return self._images[start:end], self._labels[start:end], self._rois[start:end]
      elif get_roi and get_coords:
        return self._images[start:end], self._labels[start:end], self._rois[start:end], self._coords[start:end]
      else:
        return self._images[start:end], self._labels[start:end]



  def rotational_augment(self):
    print("Rotational augment...", end="")

    self._images = rotational_augment_patches(self._images)
    self._labels = np.tile(self._labels, (9, 1))
    self._coords = np.tile(self._coords, (9, 1))
    self._image_ids = np.tile(self._image_ids, 9)
    if self._rois:
      self._rois = np.tile(self._image_ids, 9)

    self._num_images = len(self._images)
    print(self._num_images)

  def colour_augment(self):
    print("H&E stain augment...", end="")
    hed_patches = [rgb2hed(patch) for patch in self._images]
    self._images, repeats = stain_augment(hed_patches)
    del hed_patches

    # Get coords, labels, ids to match new duplicates.
    repeats = np.array(repeats).astype(np.int8)

    self._coords = np.repeat(self._coords, repeats, axis=0)
    self._labels = np.repeat(self._labels, repeats, axis=0)
    self._image_ids = np.repeat(self._image_ids, repeats, axis=0)
    if self._rois:
      self._rois = np.repeat(self._rois, repeats, axis=0)

    self._num_images = len(self._images)

    print(self._num_images)


  def read_next_patches_and_meta_test(self):
    ''' Reading the patches and meta for TEST patches into arrays. '''

    image_id = self.wsi_ids[self.wsi_index]
    print("len(wsi_ids):", len(self.wsi_ids))

    self._images = []
    self._labels = []
    self._image_ids = []
    self._rois = []
    self._coords = []

    patch_files = []
    for i in range(samples_per_patch):
        patch_files.append(str(image_id) + "_T_" + str(i) + ".h5")
    

    # patch_file = str(image_id) + "_T_" + str(patch_index) + ".h5"
    csv_file = str(image_id) + "_T.csv"

    # First read patch meta data
    with open(test_db_dir + csv_file, newline='') as metafile:
        reader = csv.reader(metafile, delimiter=' ', quotechar='|')
        for row in reader:
            # Made a mistake saving the label as an array... so deconstruct now.
            index_1 = int(row[0][4])
            new_label = [1, 0]
            if index_1:
                new_label = [0, 1]
            self._labels.append(np.array(new_label, dtype=np.int8))
            self._coords.append(np.array([int(row[1]), int(row[2])]))
            self._rois.append(int(row[3]))
            self._image_ids.append(image_id)

    flat_patches = []
    for i in range(samples_per_patch):
      flat_patches.append([])

    print("Read csv.")
    print("Labels:", np.shape(self._labels))
    print("Coords:", np.shape(self._coords))
    print("Rois:", np.shape(self._rois))
    print("Image ids:", np.shape(self._image_ids))


    # Now read patches
    for i, p in enumerate(patch_files):
        count = 0

        # Now load the images from H5 file
        file = h5py.File(test_db_dir + p,'r+')
        new_patches = np.array(file['dataset']).astype('float32')
        for patch in new_patches:
            flat_patches[i].append(np.array(patch))
            count+=1
            if count % 5000 == 0:
              print("Count...............................", count)

        del new_patches
        file.close()
        
    print("Before stacking: ", np.shape(flat_patches))
    
    # Stack patches into blocks
    for i in range(np.shape(flat_patches)[1]):
      # , flat_patches[2][i]
        self._images.append(np.concatenate((flat_patches[0][i], flat_patches[1][i]), axis=2))

    flat_patches = None
    print("After reading image_id:", image_id, " shape is:  ", np.shape(self._images))

    self._num_images = len(self._images)

    # Remember count
    self.wsi_counts[self.wsi_index] = self._num_images
    print("*** WSI index loaded:", self.wsi_index)

    
    # Check for wrap-around here and only here.
    if self.wsi_index + 1 == len(self.wsi_ids):
      self.wsi_index = 0
    else:
      self.wsi_index += 1

    print("*** Next WSI index:", self.wsi_index)


    self._images = np.array(self._images)
    self._labels = np.array(self._labels)
    self._coords = np.array(self._coords)
    self._rois = np.array(self._rois)
    self._image_ids = np.array(self._image_ids)

    # Reset index to front.
    self._index_in_epoch = 0



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


def read_k_dataset(k_id, total_k, shuffle_all=False):
  cases, gtruth = load_cases(csv_name)
  print("Cases:           ", cases)
  print("Ground truth:    ", gtruth)

  # Divided by CASE, so no patient's images will be in both valid and train sets.
  selected = np.zeros((len(cases)))
  selected[k_id::total_k] = 1
  print("Selected:        ", selected)

  wsi_ids = []
  for i, case in enumerate(cases):
    if selected[i]:
      case_file_dir = img_dir + folder_prefix + str(case) + "/"
      images = np.array([file for file in listdir(case_file_dir) if isfile(join(case_file_dir, file)) and '.svs' in file])
      
      for im in images:
        wsi_ids.append(im[:-4])

  print("WSI id's from k-set", k_id, ":", wsi_ids)
  dataset = DataSet(wsi_ids, rois=True)

  if shuffle_all:
    dataset.shuffle_all()

  return dataset

def fetch_seg_dataset(k_set_id, k, cases):

  selected = np.zeros((len(cases)))
  selected[k_set_id::k] = 1
  print("Selected:        ", selected)

  patches, coords, labels, image_ids, rois = [], [], [], [], []

  for i, case in enumerate(cases):
    if selected[i]:
      case_file_dir = img_dir + folder_prefix + str(case) + "/"
      images = np.array([file for file in listdir(case_file_dir) if isfile(join(case_file_dir, file)) and '.svs' in file])

      # images[j][:-4] image ID
      for im in images:
        # patches, coords, labels, image_ids = read_patches_and_meta_L(im[:-4], patches, coords, labels, image_ids)
        patches, coords, labels, image_ids, rois = read_patches_and_meta_test(im[:-4], patches, coords, labels, image_ids, rois)

  patches = np.array(patches)
  coords = np.array(coords)
  labels = np.array(labels)
  image_ids = np.array(image_ids)
  rois = np.array(rois)

  print("LOADED k-set:    ", k_set_id)
  print(np.shape(patches))
  print(np.shape(coords))
  print(np.shape(labels))
  print(np.shape(image_ids))

  return DataSet(patches, coords, labels, image_ids, rois)




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
      new_patches = np.array(dataset).astype('float32')
      for i, patch in enumerate(new_patches):
          patches.append(patch)

      file.close()
    except:
      # No labelled patches exist for that case.
      pass
    return patches, coords, labels, image_ids

def stain_augment(hed_patches):

  # Calculate max H, E, D range in dataset - with no colour normalisation.
  max_hed = [-2.0]*3
  min_hed = [1.0]*3
  # Keep track of how many augments are added for each index.
  # Integer at i corresponds to how many repeats for that image.
  repeats = np.ones((len(hed_patches)))

  for p in hed_patches:
    hed = np.dsplit(p, 3)
    for channel in range(3):
        # A little buffer to avoid getting unusual combinations at ends of range
        max_hed[channel] = max(np.max(hed[channel]), max_hed[channel])# - 0.0001
        min_hed[channel] = min(np.min(hed[channel]), min_hed[channel])# + 0.0001

  # print("Calculated max HED range:", max_hed)
  # print("Calculated min HED range:", min_hed)

  # Tweak the H, E, and DAB channels within the range of the dataset (one channel at a time).
  tweaked_patches = []

  for i, p in enumerate(hed_patches):
      tweaked_patches.append(p) # Always include the original
      channels = np.dsplit(p, 3)
      # Subtract and add by interval until hit min and max in any one pixel value
      interval = 0.05
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
                      repeats[i] += 1
                  else:
                      not_reached_max[ch] = False
              if not_reached_min[ch]:
                  ch_copy = np.copy(channels)
                  new_channel = np.subtract(ch_copy[ch], current)
                  if np.count_nonzero(np.where(new_channel < min_hed[ch], 1, 0)) == 0:
                      ch_copy[ch] = new_channel
                      tweaked_patches.append(np.dstack(ch_copy))
                      repeats[i] += 1
                  else:
                      not_reached_min[ch] = False
          current += interval

  del hed_patches
  new_patches = []
  for p in tweaked_patches:
    new_patches.append(hed2rgb(p))

  del tweaked_patches

  return np.array(new_patches), repeats


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









