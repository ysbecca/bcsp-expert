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

class DataSet(object):

  def __init__(self, images, labels, image_id, coords):

    self._num_images = np.array(images).shape[0]
    self._images = images

    # Boolean array versions of ID
    self._labels = labels 
    # The source image labels
    self._image_id = image_id
    # Integer labels
    # self._ids = ids 
    self._coords = coords

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def image_id(self):
    return self._image_id

  @property
  def labels(self):
    return self._labels

  @property
  def num_images(self):
    return self._num_images

  # @property
  # def ids(self):
    # return self._ids

  @property
  def set_id(self):
  	return self._set_id

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def set_images(self, images):
    self._images = images

  def set_image_cls(self, cls):
    self._image_cls = cls

  def set_labels(self, labels):
    self._labels = labels

  def set_coords(self, coords):
  	self._coords = coords

  # Shuffles all patches in the dataset object.
  def shuffle_all(self):
    if self.num_images <= 1:
        print("Cannot shuffle when", self.num_images, "images in set.")
        return

    list_all = list(zip(self._images, self._labels, self._image_id, self._coords))
    random.shuffle(list_all)
    self._images, self._labels, self._image_id, self._coords = zip(*list_all)


  def next_batch(self, batch_size, use_pseudo=False):
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


def read_datasets():
    class DataSets(object):
        pass
    dataset = DataSets()
    if is_test:
        dataset.test = fetch_dataset(turtle, -1, -1, False)
    else:
        dataset.train = fetch_dataset(turtle, set_id, total_sets, augment)
        dataset.valid = fetch_dataset(turtle, valid_id, total_sets, augment)
        if shuffle_all:
            dataset.train.shuffle_all()
            dataset.valid.shuffle_all()
    
    return dataset


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









