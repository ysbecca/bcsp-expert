
import tensorflow as tf
import cnn_helper as cn
import dataset as ds

from tensorflow.python.client import device_lib
import math
from importlib import reload
import matplotlib.pyplot as plt

from cnn_model import *
import numpy as np

from myconfig import *



total_epochs = 20
total_k = 4
roi_threshold = 0.5 # Lower threshold = higher recall, lower precision.

save_model_after = 5 # Number of epochs after which we will save every OTHER model.
show_roi_acc = 2
epochs_to_save_preds_after = 10 # Train preds, not test.


cnn_model = CNN_Model(total_k=total_k, name="24August", pretrained_epochs=0, pretrained_model=False)

print(cnn_model)
print("Model name:", cnn_model.name)
print("Total k:", cnn_model.total_k)

test_k = 4
valid_k = -1
train_ks = [0, 1, 2, 3]

for e in range(total_epochs):
	for train_k in train_ks:
		cnn_model.train_model(train_k, valid_k, epochs=1, 
									verbose=True,
									dropout_keep_prob=0.9,
									valid_interval=20,
									show_valid_acc=False, 
									train_interval=1)

		if e % show_roi_acc == 0:
			save_roi_preds = False
			if e > epochs_to_save_preds_after:
				save_roi_preds = True

			print("")
			print("ROI training accuracy based on ROI annotations:")
			roi_pred, roi_true = cnn_model.get_roi_accuracy(cnn_model.train, 
										threshold=roi_threshold,
										show_conf_matrix=False,
										verbose=True,
										save=save_roi_preds)
			print("")

	if e > save_model_after and e % 2 == 0:
		saved_model = cnn_model.save_model(train_k, epochs)
		print("========= EPOCH", e, ", saved model:", saved_model, "===========")

# Load test set and save ROI preds.
cnn_model.train = None

print("Loading test dataset, k_id:", test_k)
test_dataset = cnn_model.load_k_set(test_k, shuffle=False)


print("Calculating and saving ROI accuracy on test k:")
roi_pred, roi_true = cnn_model.get_roi_accuracy(test_dataset, 
										threshold=roi_threshold,
										show_conf_matrix=False,
										verbose=True,
										save=True)

print("...")
print("...")
print("...")

# We will only check the validation set after the epoch since still unsure about memory.
cnn_model.close_session()
print("Exiting.")


