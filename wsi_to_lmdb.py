import csv
import numpy as np
from py_wsi import *

# First, load all the cases and associated labels from the csv.

# csv_name = "exp_one_select.csv"
# cases = []
# with open(csv_name, 'r') as csvfile:
#     r = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in r:
#         cases.append(row)

# total_cases = len(cases)

# print("Cases and label meta-data read.")
# print("Total cases: " + str(len(cases)))


print("TEST: only read in first case images...")
# cases = [cases[0]]
cases = ['0001', '0002', '0003', '0004']
# Now use py_wsi to read in the WSI and sample patches.

home_dir = "/Users/ysbecca/ysbecca-projects/iciar-2018/data/WSI_xml/" #"/nobackup/sc16rsmy/bcsp-expert-cases/"
folder_prefix = "Case_"
db_location = "test_db_folder/" #"/nobackup/sc16rsmy/bcsp-lmdb/"
# db_name = "exp_one_lmdb"

patch_size = 128
level = 15
overlap = 0

for i in range(len(cases)):
	print("====================== Case " + cases[i] + " ======================")

	# Create a Turtle manager for each case -- cases[i][0]
	case_turtle = Turtle(home_dir + folder_prefix + cases[i] + "/", db_location, xml_dir=False, label_map={})
	print("Total WSI images:    " + str(case_turtle.num_files))
	print("File names:          " + str(case_turtle.files))

	# Now read the samples and store to LMDB.
	case_turtle.sample_and_store_patches(patch_size, level, overlap, load_xml=False)



print("Done sampling from all cases and saving to LMDB.")








