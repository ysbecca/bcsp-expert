
import tensorflow as tf
import cnn_helper as cn
import dataset as ds
import time
from datetime import datetime, timedelta

from tensorflow.python.client import device_lib
import math
from importlib import reload
import matplotlib.pyplot as plt

from cnn_model import *
import numpy as np

from myconfig import *



total_epochs = 10 
total_k = 4
roi_threshold = 0.5 # Lower threshold = higher recall, lower precision.

save_model_after = 1 # Number of epochs after which we will save every OTHER model.
show_roi_acc = 2
epochs_to_save_preds_after = 30 # Train preds, not test.


start_time_all = time.time()


cnn_model = CNN_Model(total_k=total_k, name="6Sept", pretrained_epochs=0, pretrained_model=False, learning_rate=1e-6)

print(cnn_model)
print("Model name:", cnn_model.name)
print("Total k:", cnn_model.total_k)

test_k = 3
train_ks = [0, 1, 2]
# Not using validation set.
valid_k = -1


print("Training ", int(total_epochs / len(train_ks)), " iterations per k.")
extra_epoch_count = 0

for e in range(total_epochs):
        start_time = time.time()
        for train_k in train_ks:
                cnn_model.train_model(train_k, valid_k, epochs=1, 
									verbose=True,
									dropout_keep_prob=0.60,
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
                        extra_epoch_count +=1
                        print("")

        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage for one epoch on ONE k-set: " + str(timedelta(seconds=int(round(time_dif)))))
        print("++++++++++++")

        if e + 1 >= save_model_after:
                saved_model = cnn_model.save_model(train_k, epochs=e)
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
extra_epoch_count +=1


print("...")
print("...")
print("...")

print("Total epochs trained: 			", total_epochs)
print("Total epochs recorded by model: 	", cnn_model.epochs)
print("Total extra epochs: 				", extra_epoch_count)
print("------------------------------")
print("Calculated total epochs: 		", cnn_model.epochs - extra_epoch_count)

# We will only check the validation set after the epoch since still unsure about memory.
cnn_model.close_session()
print("Exiting.")


end_time = time.time()
time_dif = end_time - start_time_all
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
print("++++++++++++")



print(saved_model)

