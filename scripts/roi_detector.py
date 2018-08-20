
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





cnn_model = CNN_Model(total_k=10, name="Test-CNN-Model", pretrained_epochs=0, pretrained_model=False)


print(cnn_model)
print(cnn_model.name)
print(cnn_model.total_k)


dataset = cnn_model.load_k_set(0, shuffle=False, augment=False)

# 2-class accuracy
print("Two-class accuracy based on image weak labels:")
acc, cm, cls_pred = cnn_model.get_accuracy(dataset, show_conf_matrix=False, verbose=True, get_outputs=False)
print(cm)

print("ROI accuracy based on fine ROI annotations:")

