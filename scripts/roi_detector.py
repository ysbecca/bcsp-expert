
import tensorflow as tf
import cnn_helper as cn
import dataset as ds
from tensorflow.python.client import device_lib
import math
from importlib import reload
import matplotlib.pyplot as plt

from cnn_model import *

import numpy as np


if len(sys.argv) > 1:
    case_start = int(sys.argv[1])
    case_end = int(sys.argv[2])
else:
    case_start = 0
    case_end = 50


from myconfig import *

cnn_model = CNN_Model(total_k=20, name="Test-CNN-Model", pretrained_epochs=0, pretrained_model=False)


print(cnn_model)
print(cnn_model.name)
print(cnn_model.total_k)


print("Loading k set of dataset.")
dataset = cnn_model.load_k_set(0, shuffle=False, augment=False)

print("Calculating accuracy.")

# 2-class accuracy
acc, cm, cls_pred = cnn_model.get_accuracy(dataset, show_conf_matrix=False, verbose=True, get_outputs=False)
print("Two-class accuracy based on image weak labels:")
print(cm)

print("ROI accuracy based on fine ROI annotations:")

