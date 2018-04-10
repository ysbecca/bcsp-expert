import csv
import numpy as np
from py_wsi import *

# First, load all the cases and associated labels from the csv.

csv_name = "exp_one_select.csv"
cases = []
with open(csv_name, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in r:
        cases.append(row)

total_cases = len(cases)

print("Cases and label meta-data read.")
print("Total cases: " + str(len(cases)))


print("TEST: only read in first case images...")
cases = [cases[0]]

# Now use py_wsi to read in the WSI and sample patches.

home_dir = "/nobackup/sc16rsmy/bcsp-expert-cases/"
folder_prefix = "Case_"
db_location = "/nobackup/sc16rsmy/bcsp-lmdb/"
db_name = "exp_one_lmdb"

patch_size = 256
level = 15
overlap = 0.5

for i in range(total_cases):
	print("===== Case " + str(cases[i][0]) + " =====")

	# Create a Turtle manager for each case
	case_turtle = Turtle(home_dir + folder_prefix + cases[i][0], db_location, db_name, xml_dir=False, label_map={})
	print("Total WSI images:    " + str(turtle.num_files))
	print("LMDB name:           " + str(turtle.db_name))
	print("File names:          " + str(turtle.files))

	# Now read the samples and store to LMDB.
	case_turtle.sample_and_store_patches(patch_size, level, overlap, load_xml=False)



print("Done sampling from WSI and saving to LMDB.")








