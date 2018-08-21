
import tensorflow as tf
import cnn_helper as cn
import dataset as ds
from tensorflow.python.client import device_lib
import math
from importlib import reload
import matplotlib.pyplot as plt

from myconfig import *
from cnn_model import *

import numpy as np



cnn_model = CNN_Model(total_k=2, name="Test-CNN-Model", pretrained_epochs=0, pretrained_model=False)

print(cnn_model)
print("Model name:", cnn_model.name)
print("Total k:", cnn_model.total_k)


# Training a model
print("TRAINING ----------------- ")
cnn_model.train_model(0, 1, epochs=1, 
							verbose=True, 
							dropout_keep_prob=0.9,
							valid_interval=2,
							show_valid_acc=False, 
							train_interval=1)


# Computing ROI accuracy
print("ROI accuracy based on fine ROI annotations:")
roi_pred, roi_true = cnn_model.get_roi_accuracy(cnn_model.train, 
							threshold=0.5,
							show_conf_matrix=False,
							verbose=True,
							save=True)

# Saving the model
# cnn_model.save_model(1, epochs=0)

print("Exiting.")


