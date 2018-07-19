'''

Save all the config for patch sampling and loading here.

@ysbecca

'''


# To read the case numbers selected
csv_name = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_cases.csv"

# General high directory containing all the cases
img_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/wsi_samples/"

# Where to save the created h5 and csv files
test_db_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_db/"

# Where to find the JSON annotation files
json_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/wsi_json/"

folder_prefix = "Case_"
patch_size = 128
level = 13
pixel_overlap = 0

# For image quality downsampling.
downsample_factor = 8

new_patch_dim = int(patch_size / downsample_factor)

# Downsampling annotated regions for quickly labeling patches
annotation_downsample = 10 
label_map = {"EP": 0,
             "SM": 1,
             "U": -1} # Unknown