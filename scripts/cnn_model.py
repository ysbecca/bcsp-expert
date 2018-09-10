
# coding: utf-8

# ## Network model class 
# 
# Can be adjusted for future use.
# Author @ysbecca

# In[1]:

import tensorflow as tf
import cnn_helper as cn
import dataset as ds

from tensorflow.python.client import device_lib
import math
from importlib import reload
import matplotlib.pyplot as plt

from myconfig import *
import time
from datetime import datetime, timedelta

import numpy as np
import csv



# In[2]:

# reload(cn)
# reload(ds)


# All model architecture parameters defined in a config file.

# In[3]:

# In[14]:

class CNN_Model():
    
    def __init__(self, 
                 total_k,
                 name="CNN Model",
                 learning_rate=1e-4,
                 pretrained_epochs=0,
                 pretrained_model=False,
                 weights_array=[],
                ):
        ''' Initalises all the network parameters and starts a TF session. '''
        
        self.total_k = total_k
        self.weights_array = weights_array
        self.epochs = pretrained_epochs
        
        # Load pre-trained model, if provided.
        self.pretrained_model = pretrained_model
    
        self.name = name
    
        self.valid = None
        self.train = None
        self.epochs = 0
        
        # Initialises the TF session and loads pretrained model if it exists.
        
        # Build tensors.
        self.x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="x")
        # tf.reshape(x, [-1, img_size, img_size, num__channels])
        self.x_image = tf.reshape(self.x, [-1, img_size, img_size, num_channels], name="x_image")
        
        #self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        
        self.y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")
        self.y_true_cls = tf.argmax(self.y_true, dimension=1, name="y_true_cls")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob") # Dropout control.
        self.gpus = self.__get_gpus()
        
        
        if len(self.gpus) > 0:
            # Remember that this adds significant latency for CPU<->GPU copying of shared variables.
            # 2 GPU's is enough to get a good balance between speedup and minimal latency (12 GB on k80 nodes)
            self.y_pred, self.y_pred_cls, self.cost = self.make_parallel(
                                                    self.model, 
                                                     x_image_=self.x_image, 
                                                     y_true_=self.y_true)
                                                     #weights_=self.weights)
            total_batches = len(self.gpus)
        else:
            # CPU-only version
            self.y_pred, self.y_pred_cls, self.cost = self.model(x_image_=self.x_image, 
                                                       y_true_=self.y_true)
                                                       #weights_=self.weights)
            total_batches = 1

            
        self.train_batch_size = train_batch_size * total_batches
        self.test_batch_size = test_batch_size * total_batches
    
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=
                learning_rate).minimize(self.cost, 
                                        colocate_gradients_with_ops=True)

        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        if len(self.gpus) > 0: # Log GPU/CPU placement to the terminal if using GPU's.
            self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        else:
            self.session = tf.Session()

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        # Load pretrained model
        if self.pretrained_model:
            self.restore_model(self.pretrained_model)


    
    
    def model(self, x_image_, y_true_):
        ''' Defines the model which can be copied to each GPU device being used. '''

        model = x_image_
        num_filters_ = [num_channels] + num_filters
        print("Num Filters:", num_filters_)
        print("Filter sizes:", filter_sizes)

        for i in range(num_layers):
            print("i:", i)
            print("Input channels:", num_filters_[i])
            print("Filter sizes:", filter_sizes[i])
            
            model = cn.new_conv_layer(input=model,
                                          num_input_channels=num_filters_[i],
                                          filter_size=filter_sizes[i],
                                          num_filters=num_filters_[i+1],
                                          use_pooling=True,
                                          max_pool_size=max_pools[i],
                                          use_relu=relu[i])
            if i > 0:
                # Don't dropout the input layer
                model = tf.nn.dropout(model, self.keep_prob)
                print(model)


        model, num_fc_features = cn.flatten_layer(model)
        print(num_fc_features)

        if len(fc_sizes) > 0:
            fc_sizes_ = [num_fc_features] + fc_sizes

            # The flattened features will then be fed into the fully-connected layers.
            for i in range(len(fc_sizes)):
                model = cn.new_fc_layer(input=model,          
                                     num_inputs=fc_sizes_[i],
                                     num_outputs=fc_sizes_[i+1],
                                     use_relu=True)
                #model = tf.nn.dropout(model, self.keep_prob)

            num_fc_features = fc_sizes_[-1]

        model = cn.new_fc_layer(input=model,          
                                 num_inputs=num_fc_features,
                                 num_outputs=num_classes,
                                 use_relu=True)

        y_pred = tf.nn.softmax(model)
        y_pred_cls = tf.argmax(model, dimension=1)
        cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=model, onehot_labels=y_true_))
                                                              #, weights=weights_))
        return y_pred, y_pred_cls, cost
        
    
    def train_model(self, k, 
                        valid_k, 
                        epochs, 
                        verbose=True, 
                        valid_interval=100, 
                        dropout_keep_prob=0.9, 
                        show_valid_acc=True, 
                        train_interval=10):
        ''' Train the network for the given number of epochs. '''
        start_time = time.time()
        
        # Load k training set and valid_k set.
        if not self.train:
            self.load_train_dataset(k)
        
        if show_valid_acc and not self.valid:
            self.load_valid_dataset(valid_k)
        
        # Optimise.
        acc = self.optimize(self.train, 
                            epochs,
                            dropout_keep_prob, 
                            verbose, 
                            show_valid_acc, 
                            valid_interval=valid_interval,
                            train_interval=train_interval)
        
        end_time = time.time()
        time_dif = end_time - start_time
        if verbose:
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    
    def load_valid_dataset(self, valid_k):
        self.valid = ds.read_k_dataset(valid_k, self.total_k, shuffle_all=True)
        
    def load_train_dataset(self, train_k):
        self.train = ds.read_k_dataset(train_k, self.total_k, shuffle_all=True)
    

    def get_roi_accuracy(self, dataset, threshold=0.5, 
                            show_conf_matrix=True,
                            verbose=True,
                            save=False):
        ''' Computes the ROI accuracy given threshold and based on dataset.roi 
        '''
        if dataset.wsi_index != 1:
            dataset.reset_all()

        if not dataset:
            print("OUPS! No dataset loaded.")
            return None

        # Iterate through dataset and compare values above threshold for matching class
        # with whether it is within the ROI or not.
        cls_pred, cls_true, roi_true, coords = [], [], [], []
        count = 0        

        start_epoch = dataset.epoch_count

        values = [0, 0]
        while dataset.epoch_count < start_epoch + 1:
            x_batch, y_true_batch, roi_true_batch = dataset.next_batch(self.test_batch_size, 
                            get_roi=True,
                            stop_at_epoch=True)


            # Compensate for uneven batch size by replicating last image.
            x_batch, y_true_batch, roi_true_batch  = self.adjust_batch_size([x_batch, y_true_batch, roi_true_batch])
            print("**", len(x_batch))

            x_batch = x_batch.reshape(len(x_batch), img_size_flat)
            feed_dict = {self.x: x_batch, self.y_true: y_true_batch, self.keep_prob: 1.0}
            
            outs = self.session.run(self.y_pred, feed_dict=feed_dict)
            for j, out in enumerate(outs):
                cls_pred.append(out)
                cls_true.append(y_true_batch[j])
                roi_true.append(roi_true_batch[j])
            
            count += len(x_batch)

        # Calculate how many cls_preds are 1) correct AND 2) above the threshold.
        cls_pred = np.array(cls_pred)
        cls_true = np.array(cls_true)
        roi_true = np.array(roi_true)

        correct_cls = np.array(np.argmax(cls_pred, axis=1) == np.argmax(cls_true, axis=1))
        # print("Correct class:   ", correct_cls)
        # print("True ROI:        ", roi_true)
        
        selected_roi = np.where(correct_cls, np.where(np.max(cls_pred, axis=1) > threshold, 1, 0), 0)
        # print("Found ROI:       ", selected_roi)

        correct_roi = np.array(selected_roi == roi_true)
        # print("Correct ROI:     ", correct_roi)

        correct_sum = correct_roi.sum()
        acc = float(correct_sum) / count

        cm = cn.plot_confusion_matrix(correct_roi, cls_pred=selected_roi, show_plt=show_conf_matrix)
        if len(cm) > 1: # If both arrays all 0 or all 1, then only 1-dim conf matrix.
            tn, fp, fn, tp = cm.ravel()
            recall = float(tp) / (tp + fn)
        else:
            recall = acc

        print("Conf matrix:")
        print(cm)

        if verbose:
            msg = "ROI accuracy: {0:.1%} ({1} / {2})... Recall: {0:.1%}"
            print(msg.format(acc, correct_sum, count, recall))

        if save:
            self.save_preds(dataset, selected_roi, cls_pred)

        return np.array(selected_roi), np.array(roi_true)

    def adjust_batch_size(self, lists):
        ''' Makes sure batch size is evenly divisible by number of gpus. '''
        mod = len(lists[0]) % len(self.gpus)
        to_add = len(self.gpus) - mod

        if mod == 0:
            return lists

        print("mod", mod, "to_add", to_add)

        print(np.shape(lists[0]))
        print(np.shape(lists[1]))

        third_list = False
        if len(lists) > 2:
            third_list = True
            print(np.shape(lists[2]))
            c = lists[2].tolist()

        a, b = lists[0].tolist(), lists[1].tolist() 
        
        for n in range(to_add):
            a.append(lists[0][-1])
            b.append(lists[1][-1])
            if third_list:
                c.append(lists[2][-1])

        a, b = np.array(a), np.array(b)
        print(np.shape(a))
        print(np.shape(b))
        
        if third_list:
            c = np.array(c)
            print(np.shape(c))

        if third_list:
            return a, b, c
        else:
            return a, b

    def save_preds(self, dataset, selected_roi, cls_pred):
        ''' Any wsi for which the count is unknown will end up saving an empty pred file. '''

        print("len selected_roi", len(selected_roi))

        print("dataset.wsi_index", dataset.wsi_index)
        print("dataset.wsi_counts", dataset.wsi_counts)

        curr = 0
        for i, count in enumerate(dataset.wsi_counts):
            
            csv_name = str(dataset.wsi_ids[i]) + "_preds.csv"

            with open(preds_dir + csv_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for j in range(curr, curr+count):
                    row = [selected_roi[j], cls_pred[j][0], cls_pred[j][1]]
                    writer.writerow(row)

            curr += count

    def get_accuracy(self, dataset, 
                        show_conf_matrix=True, 
                        verbose=True, 
                        get_outputs=False,
                        acc_type=""):
        ''' Computes validation/test accuracy; used during training and as a standalone function. Batch
            size is test_batch_size since forward pass only. '''
        
        if not dataset:
            print("OUPS! No dataset loaded.")
            return None
            

        if dataset.wsi_index != 0:
            dataset.reset_all()

        cls_pred = []
        cls_true = []
        count = 0

        while dataset.epoch_count < 1: # TODO FIX

            x_batch, y_true_batch = dataset.next_batch(test_batch_size, 
                                                    stop_at_epoch=True)

            

            x_batch = x_batch.reshape(len(x_batch), img_size_flat)
            
            #weights_ = np.ones((curr_batch_size))
            feed_dict = {self.x: x_batch, self.y_true: y_true_batch, self.keep_prob: 1.0}
            
            if get_outputs: # NOT thresholded outputs
                outs = self.session.run(self.y_pred, feed_dict=feed_dict)
            else:
                outs = self.session.run(self.y_pred_cls, feed_dict=feed_dict)
            
            for j, out in enumerate(outs):
                cls_pred.append(out)
                cls_true.append(y_true_batch[j])
            count += len(x_batch)


        # If getting Non-thresholded outputs, return with labels
        if get_outputs:
            return outs, cls_true

        cls_true = [np.argmax(c) for c in cls_true]

        # Create a boolean array whether each image is correctly classified.
        correct = np.array(np.array(cls_true) == np.array(cls_pred))
        correct_sum = correct.sum() 
        
        acc = float(correct_sum) / count

        if verbose:
            msg = acc_type + " Accuracy: {0:.1%} ({1} / {2})"
            print(msg.format(acc, correct_sum, count))
        
        cm = cn.plot_confusion_matrix(cls_true, cls_pred=cls_pred, show_plt=show_conf_matrix)
        # print("Total patches loaded in dataset:", count)

        dataset.reset_epoch_count()
        return acc, cm, cls_true, cls_pred
        
    
    def save_outputs(self, k, outputs):
        ''' Saves softmax outputs to a file. '''
        
        #TOOD save outputs into specified directory in a good format.
        print("Oups! Not implemented yet.")
        return -1
    
    def load_k_set(self, k, shuffle):
        ''' Loads the kth set without setting the self.train or self.valid variables. '''
        return ds.read_k_dataset(k, self.total_k, shuffle_all=shuffle)
    
    def __get_gpus(self):
        gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        if len(gpus) == 0:
            return ['cpu']
        else:
            return gpus
    
    def make_parallel(self, fn, **kwargs):
        ''' Splits the model across available GPUs. '''
        in_splits = {}
        for k, v in kwargs.items():
            in_splits[k] = tf.split(v, len(self.gpus))

        y_pred_split, y_pred_cls_split, cost_split = [], [], []
        for i in range(len(self.gpus)):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                    y_pred_, y_pred_cls_, cost_ = fn(**{k : v[i] for k, v in in_splits.items()})
                    y_pred_split.append(y_pred_)
                    y_pred_cls_split.append(y_pred_cls_)
                    cost_split.append(cost_)

        return tf.concat(y_pred_split, axis=0), tf.concat(y_pred_cls_split, axis=0),                 tf.stack(cost_split, axis=0)
    
    # Saving and restoring models.
    def save_model(self, k, epochs=0):
        model_name = self.name + '-' + datetime.now().strftime("%Y-%m-%d-%H-%M") + "-" + str(k) + "-" + str(epochs)
        self.saver.save(self.session, checkpoints_dir + model_name)
        print("CNN Model saved for k=", k, ":", model_name)
        
        return model_name

    def restore_model(self, model_name):
        output = self.saver.restore(sess=self.session, save_path=checkpoints_dir + model_name)
        print(output)

    def optimize(self, dataset, 
                        epochs,
                        dropout_keep_prob, 
                        verbose, 
                        show_valid_acc, 
                        valid_interval, 
                        train_interval):
        ''' Optimises for the given number of iterations by batches. '''
        
        start_epoch = dataset.epoch_count
        # print("Starting training at epoch:", start_epoch)

        while dataset.epoch_count < start_epoch + epochs:
            # print("Epoch count:", dataset.epoch_count)

            x_batch, y_true_batch = dataset.next_batch(train_batch_size)
            #x_batch, y_true_batch = self.adjust_batch_size([x_batch, y_true_batch])
            print(np.shape(x_batch))
            x_batch = x_batch.reshape(len(x_batch), img_size_flat)

            #weights_ = generate_batch_weights(y_true_pseudo)
            feed_dict_train = {self.x: x_batch, self.y_true: y_true_batch, self.keep_prob: dropout_keep_prob}

            if dataset.epoch_count % valid_interval == 0 and show_valid_acc:
                acc, _, _, _ = self.get_accuracy(self.valid, show_conf_matrix=False, acc_type="Validation")
                msg = "Validation Accuracy:    {0:>6.1%}  Epochs: {1}"
                print(msg.format(acc, dataset.epoch_count))

            self.session.run(self.optimizer, feed_dict=feed_dict_train)

            if dataset.epoch_count % train_interval == 0:
                acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
                msg = "Training Accuracy:      {0:>6.1%}  Epochs:    {1}"
                print(msg.format(acc, dataset.epoch_count))        


        self.epochs += epochs

        return acc


    def close_session(self):
        print("Closing TensorFlow session.")
        self.session.close()

    def reset_epochs(self):
        self.epochs = 0




