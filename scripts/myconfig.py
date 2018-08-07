'''

Save all the config for patch sampling and loading here.

@ysbecca

'''


# To read the case numbers selected
# csv_name = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/data/training_cases_only.csv"
csv_name = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/data/test_cases_only.csv"
# csv_name = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_cases.csv"

# General high directory containing all the cases
img_dir = "/nobackup/sc16rsmy/bcsp-expert-cases/"
# img_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/wsi_samples/"

# Where to save the created h5 and csv files
test_db_dir = "/nobackup/sc16rsmy/bcsp-expert-h5/"
# test_db_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/temp_db/"

# Where to find the JSON or XML annotation files
# json_dir = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/annotations/training_set/"
json_dir = "/home/ufaserv1_k/sc16rsmy/bcsp-expert/annotations/test_set/"
# json_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/data/wsi_json/"
# xml_dir = "/Users/ysbecca/ysbecca-projects/bcsp-expert/annotations/test_set/"

folder_prefix = "Case_"
patch_sizes = [512, 1024, 2048, 4096]
level = 0
pixel_overlap = 0
samples_per_patch = len(patch_sizes)


# For image quality downsampling.
downsamples = [0, 2, 4, 8]

# new_patch_dim = int(patch_size / downsample_factor)

# Downsampling annotated regions for quickly labeling patches
annotation_downsample = 10

label_map = {"EP": 0,
             "SM": 1,
             "U": -1} # Unknown
