'''

Save all the config for patch sampling and loading here.

@ysbecca

'''
import sys


# For selectively reading one patch type alone.
patch_index = 1

gpus = 4

case_start = 0
case_end = 2

# To read the case numbers selected
# csv_name = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/data/training_cases_only.csv" 
csv_name = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/data/test_cases_only.csv"
#csv_name = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_cases.csv"

# General high directory containing all the cases
img_dir = "/nobackup/sc16rsmy/bcsp-expert-cases/"
#img_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/wsi_samples/"

# Where to save the created h5 and csv files
test_db_dir = "/nobackup/sc16rsmy/bcsp-expert-h5/"
#test_db_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_db/"

# Where to find the JSON or XML annotation files
# json_dir = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/annotations/training_set/"
xml_dir = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/annotations/test_set/"
# json_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/wsi_json/"
#xml_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/annotations/test_set/"


# Directories
model_dir = "/nobackup/sc16rsmy/bcsp-expert-cases/checkpoints-bcsp/"
checkpoints_dir = model_dir

# ROI and pred outputs
preds_dir = "/nobackup/sc16rsmy/bcsp-expert-cases/preds-bcsp/"

# print("Case start, end: ", case_start, case_end)

folder_prefix = "Case_"
base_patch_size = 256
patch_sizes = [512, 1024, 2048]
level = 0
pixel_overlap = 0
samples_per_patch = len(patch_sizes)


chunk_size = 2000

# For image quality downsampling.
downsamples = [2, 4, 8]

# new_patch_dim = int(patch_size / downsample_factor)

# Downsampling annotated regions for quickly labeling patches
annotation_downsample = 10

label_map = {"EP": 0,
             "SM": 1,
             "U": -1} # Unknown


# Input parameters
img_size = 256
num_channels = 9

img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)

# Convolutional layer parameters
filter_sizes = [7, 5, 3, 1]
num_filters = [16, 16, 16, 32]
num_layers = len(filter_sizes)
max_pools = [2, 2, 2, 2]
relu = [1, 1, 1, 1]

fc_sizes = [256, 128]
num_classes = 2

# Training params
train_batch_size = 8
test_batch_size = 6            
