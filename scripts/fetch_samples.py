'''
@author ysbecca

Functions which take URLs to images from a .csv and fetch sample patches from them given 
a patch size, location, quality, and zoom.

'''

import numpy as np
import csv
import cv2
from skimage import draw
from PIL import Image
import urllib.request
import io

DATASET_CSV_FILE = "./eindhoven_dataset.csv"


def get_image_ids(urls_array):
	''' Given an array of all the image url's, parses out the image ids and returns them in a
		an array. '''
	image_ids = []
	for url in urls_array:
		image_id = re.search("(\w+)(\.\w+)+(?!.*(\w+)(\.\w+)+)", url).group(1)
		image_ids.append(int(image_id))
	return image_ids


def load_urls(path):
	''' Loads all the URLs to images into a list from the csv file. '''
	urls = []

	with open(path, newline='') as csv_file:
	    reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
	    for row in reader:
	        urls.append(', '.join(row))

	return urls

def fetch_patch(url, x, y, patch_x, patch_y, zoom, quality):
	''' Given a URL for an image, extract the path with the given parameters. '''

	constructed_url = (url + "?" + str(x) + "+" + str(y) + "+" + str(patch_x) + "+" + str(patch_y) + "+" + str(zoom) + "+" + str(quality))
	file = read_file_from_url(constructed_url)

	patch = Image.open(file)
	return np.array(patch)

def mask_image(image_size, coords, stride):
    adj_x = int(np.floor(image_size[1]/stride))
    adj_y = int(np.floor(image_size[0]/stride))
    formatted_coords = np.array(np.floor(np.divide(coords, stride)), 'int32')
    
    col_coords, row_coords =  np.hsplit(formatted_coords, 2)
    row_coords = np.ravel(row_coords)
    col_coords = np.ravel(col_coords)
    
    fill_row_coords, fill_col_coords = draw.polygon(row_coords, col_coords, (adj_x, adj_y))

    mask = np.zeros((adj_x, adj_y), dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = 1
    return mask

def draw_boundary(image, image_size, coords, stride):
	''' Draws the annotation on top of the image (thumbnail recommended!)
	'''
	adjusted_x = int(np.ceil(image_size[1]/stride))
	adjusted_y = int(np.ceil(image_size[0]/stride))
	formatted_coords = np.array(np.ceil(np.divide(coords, stride)), 'int32')

	# Assuming 3 channels (RGB) - set all outside region to black
	points = formatted_coords.reshape((-1,1,2))
	cv2.polylines(image, np.int32([points]), True, (0, 255, 0), 3)
	return image

def fetch_thumbnail(url):
	''' Returns the thumbnail of the image. '''

	file = read_file_from_url(url + str("?-1"))
	thumbnail = Image.open(file)
	
	return np.array(thumbnail)

def fetch_size(url):
	''' Returns information about the image x, y dimensions, etc. Returns [width, height]. '''
	file = read_file_from_url(url + str("?INFO"))
	string_info = file.getvalue().decode("utf-8") 
	info = string_info.split("|")

	return int(info[0]), int(info[1])

def read_file_from_url(built_url):
	''' Reads the file at the given URL and returns it. '''
	with urllib.request.urlopen(built_url) as url:
	    file = io.BytesIO(url.read())
	return file

def reduced_to_original(x, y, stride):
	''' Conversion between dimension-reduced coordinates and original, based on stride. '''
	return int((x - 1)*stride), int((y - 1)*stride)

def original_to_reduced(x, y, stride):
	''' Conversion between original coordinates and dimension-reduced ones, based on stride. '''
	return int(np.ceil(x / stride)), int(np.ceil(y / stride))



''' NOTE: if stride <= patch_size/2 then we are OK with dimension reduction for patch selection, as
	patches will be fully inside or outside the ROI. We ignore all impure patches (border patches).
'''



def main():
	''' Parses a test XML and displays first 10 coordinates of first region. '''
	print("TEST: running fetch-samples script.")
	print("--------")
	# First load the URLs from the CSV
	test_urls = load_urls(images_csv_test_path)
	print("First 5 URL's retrieved from CSV:")
	print("--------")
	print(test_urls[0:5])

	x = 20000
	y = 20000
	patch_x = 500
	patch_y = 500
	zoom = 1
	quality = 100
	print("--------")
	
	# Now retrieve a single patch at the given location and params from the first test image.
	test_patch = fetch_patch(test_urls[0], x, y, patch_x, patch_y, zoom, quality)
	# test_patch.show()

	# Print size
	print("--------")
	image_size = fetch_size(test_urls[0])
	print("Size of entire image:")
	print(image_size)

	stride = 4
	original_size = (400, 300)
	test_coords = np.array([[20, 20], [275, 50], [200, 150], [100, 200], [30, 120]])

	print("--------")
	print("TEST EXAMPLE: generating a mask for image.")
	print("Original size: " + str(original_size))
	print("Stride for dim. reduction: " + str(stride))

	print("--------")
	# print("Original annotation coords: ")
	# print(test_coords)
	# print("Reduced annotation coords: ")

	mask = mask_image(original_size, test_coords, stride)
	ysb.show_image(mask)

if __name__ == "__main__":
    main()


