import csv
import numpy as np
from py_wsi import *
import time
from datetime import timedelta

# First, load all the cases and associated labels from the csv.

<<<<<<< HEAD
# csv_name = "exp_one_select.csv"
# cases = []
# with open(csv_name, 'r') as csvfile:
#     r = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in r:
#         cases.append(row)
=======
start_time = time.time()

csv_name = "exp_one_select.csv"
cases = []
with open(csv_name, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in r:
        cases.append(row)
>>>>>>> ca8236d3023c7610dd34135773d39e0738dd3802

# total_cases = len(cases)

# print("Cases and label meta-data read.")
# print("Total cases: " + str(len(cases)))

<<<<<<< HEAD

print("TEST: only read in first case images...")
# cases = [cases[0]]
cases = ['0001', '0002', '0003', '0004']
=======
start_range = 0
end_range = 5
print("Range of cases being processed: " + str(start_range) + ":" + str(end_range))
cases = cases[start_range:end_range]

>>>>>>> ca8236d3023c7610dd34135773d39e0738dd3802
# Now use py_wsi to read in the WSI and sample patches.

home_dir = "/Users/ysbecca/ysbecca-projects/iciar-2018/data/WSI_xml/" #"/nobackup/sc16rsmy/bcsp-expert-cases/"
folder_prefix = "Case_"
db_location = "test_db_folder/" #"/nobackup/sc16rsmy/bcsp-lmdb/"
# db_name = "exp_one_lmdb"

patch_size = 128
level = 15
overlap = 0

<<<<<<< HEAD
for i in range(len(cases)):
	print("====================== Case " + cases[i] + " ======================")

	# Create a Turtle manager for each case -- cases[i][0]
	case_turtle = Turtle(home_dir + folder_prefix + cases[i] + "/", db_location, xml_dir=False, label_map={})
	print("Total WSI images:    " + str(case_turtle.num_files))
=======
for i in range(total_cases):
	print("===== Case " + str(cases[i][0]) + " =====")

	# Create a Turtle manager for each case
	case_turtle = Turtle(home_dir + folder_prefix + cases[i][0] + "/", db_location, db_name, xml_dir=False, label_map={})
	print("Total WSI images:    " + str(case_turtle.num_files))
	print("LMDB name:           " + str(case_turtle.db_name))
>>>>>>> ca8236d3023c7610dd34135773d39e0738dd3802
	print("File names:          " + str(case_turtle.files))

	# Now read the samples and store to LMDB.
	case_turtle.sample_and_store_patches(patch_size, level, overlap, load_xml=False)



print("Done sampling from all cases and saving to LMDB.")

end_time = time.time()
time_elapsed = str(timedelta(seconds=int(round(end_time - start_time))))
print("Time usage: " + time_elapsed)

'''
print("Writing time and range to file.")

path = "record_" + str(start_range) + "_" + str(end_range) + ".txt"
record_file = open(path,'w')
record_file.write("Range: " + str(start_range) + ":" + str(end_range) + "\n")
record_file.write("Time elapsed: " + time_elapsed + "\n")

record_file.close()
'''

