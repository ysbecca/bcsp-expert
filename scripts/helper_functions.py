import time
from datetime import timedelta
import matplotlib.pyplot as plt


import csv


from myconfig import case_start, case_end

# Loads csv cases from file
def load_cases(csv_name):
    cases, gtruth = [], []
    with open(csv_name, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in r:
            cases.append(row[0])
            gtruth.append(row[1])

    return cases[case_start:case_end], gtruth[case_start:case_end]

# Helper timing functions.
def start_timer():
	return time.time()

def end_timer(start_time):
	end_time = time.time()
	print(str(timedelta(seconds=int(round(end_time - start_time)))))



# Visualising patches.
def show_labeled_patches(images, clss):
    fig = plt.figure(figsize=(20, 10))
    data = images[:50]
    labels = clss[:50]

    for i, image in enumerate(data):
        plt.subplot(5, 10, i+1)
        plt.imshow(image)
        plt.title(str(labels[i]))
        plt.axis("off")

    plt.show()
