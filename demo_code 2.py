def next_batch_threeview(num, view1_data, view2_data, view3_data, labels):
    idx = np.arange(0, len(view1_data))
    np.random.shuffle(idx)
    idx = idx[:num]
    view1_data_shuffle = [view1_data[i] for i in idx]
    view2_data_shuffle = [view2_data[i] for i in idx]
    view3_data_shuffle = [view3_data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(view1_data_shuffle), np.asarray(view2_data_shuffle), np.asarray(view3_data_shuffle), np.asarray(labels_shuffle)
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
# from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib import rnn
import os
import h5py
import numpy as np
import scipy.io as sio
tf.reset_default_graph()


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#####################
### load rgb data ###
#####################
print('start to load rgb data')

train_d = h5py.File('your_data')
rgb_train_data = train_d['your_data']

train_l = h5py.File('your_label')
rgb_train_label= train_l['your_label']
print(np.shape(rgb_train_data))
print(np.shape(rgb_train_label))
print('finish loading rgb training')


test_d = h5py.File('your_data')
rgb_test_data = test_d['your_data']

test_l = h5py.File('your_label')
rgb_test_label= test_l['your_label']
print(np.shape(rgb_test_data))
print(np.shape(rgb_test_label))
print('finish loading rgb testing')

#####################
### load dep data ###
#####################
print('start to load depth data')
train_d = h5py.File('your_data')
depth_train_data = train_d['your_data']

train_l = h5py.File('your_label')
depth_train_label= train_l['your_label']
print(np.shape(depth_train_data))
print(np.shape(depth_train_label))
print('finish loading depth training')

test_d = h5py.File('your_data')
depth_test_data = test_d['your_data']

test_l = h5py.File('your_label')
depth_test_label= test_l['your_label']
print(np.shape(depth_test_data))
print(np.shape(depth_test_label))
print('finish loading depth testing')

#####################
### load ske data ###
#####################
print('start to load skeleton data')
train = h5py.File('your_data')
skeleton_train_data = train['your_data']
skeleton_train_label= train['your_label']
print(np.shape(skeleton_train_data))
print(np.shape(skeleton_train_label))
print('finish loading skeleton training')
test = h5py.File('your_data')
skeleton_test_data = test['your_data']
skeleton_test_label= test['your_label']
print(np.shape(skeleton_test_data))
print(np.shape(skeleton_test_label))
print('finish loading skeleton testing')

################
### rgb LSTM ###
################


# Network Parameters
rgb_input_size = np.shape(rgb_train_data)[2] # feature dimension
rgb_timestep_size = np.shape(rgb_train_data)[1] # timesteps
rgb_hidden_size = 128 # hidden layer num of features
class_num = 20 
rgb_layer_num = 1

# LSTM Model
rgb_X = tf.placeholder(tf.float32, [None, rgb_timestep_size, rgb_input_size], name = 'rgb_X')

rgb_y = tf.placeholder(tf.float32, [None, class_num], name = 'rgb_y')
rgb_keep_prob = tf.placeholder(tf.float32, name = 'rgb_keep_prob')

def get_a_cell(lstm_size, keep_prob, cell_name):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, name = cell_name)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop

rgb_mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(rgb_hidden_size, rgb_keep_prob, 'rgb_lstm') for _ in range(rgb_layer_num)])

rgb_outputs, rgb_state = tf.nn.dynamic_rnn(rgb_mlstm_cell, inputs=rgb_X, dtype=tf.float32)
# last one
rgb_h_state = rgb_outputs[:, -1, :]
#
rgb_W = tf.Variable(tf.truncated_normal([rgb_hidden_size, class_num], stddev=0.1), dtype=tf.float32)
rgb_bias = tf.Variable(tf.constant(1.0, shape=[class_num]), dtype=tf.float32)


rgb_y_pre = tf.nn.softmax(tf.matmul(rgb_h_state, rgb_W) + rgb_bias)

rgb_cross_entropy = tf.reduce_mean(tf.square(rgb_y - rgb_y_pre))

rgb_result = tf.argmax(rgb_y_pre, 1)
rgb_target = tf.argmax(rgb_y, 1)
rgb_correct_prediction = tf.equal(tf.argmax(rgb_y_pre, 1), tf.argmax(rgb_y, 1))
rgb_accuracy = tf.reduce_mean(tf.cast(rgb_correct_prediction, "float"))
rgb_confusion_matrix = tf.contrib.metrics.confusion_matrix(rgb_result, rgb_target, num_classes= None, dtype = tf.int32, name=None, weights= None)
##

################
### dep LSTM ###
################
depth_input_size = np.shape(depth_train_data)[2] #
depth_timestep_size = np.shape(depth_train_data)[1] # timesteps
depth_hidden_size = 128 # hidden layer num of features
class_num = 20 # 
depth_layer_num = 1

# LSTM Model
depth_X = tf.placeholder(tf.float32, [None, depth_timestep_size, depth_input_size])

depth_y = tf.placeholder(tf.float32, [None, class_num])
depth_keep_prob = tf.placeholder(tf.float32)



depth_mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(depth_hidden_size, depth_keep_prob, 'depth_lstm') for _ in range(depth_layer_num)])

depth_outputs, depth_state = tf.nn.dynamic_rnn(depth_mlstm_cell, inputs=depth_X, dtype=tf.float32)#, initial_state=init_state, time_major=False)
# last one
depth_h_state = depth_outputs[:, -1, :]

depth_W = tf.Variable(tf.truncated_normal([depth_hidden_size, class_num], stddev=0.1), dtype=tf.float32)
depth_bias = tf.Variable(tf.constant(1.0, shape=[class_num]), dtype=tf.float32)

depth_y_pre = tf.nn.softmax(tf.matmul(depth_h_state, depth_W) + depth_bias)

depth_cross_entropy = tf.reduce_mean(tf.square(depth_y - depth_y_pre))

depth_result = tf.argmax(depth_y_pre, 1)
depth_target = tf.argmax(depth_y, 1)
depth_correct_prediction = tf.equal(tf.argmax(depth_y_pre, 1), tf.argmax(depth_y, 1))
depth_accuracy = tf.reduce_mean(tf.cast(depth_correct_prediction, "float"))
depth_confusion_matrix = tf.contrib.metrics.confusion_matrix(depth_result, depth_target, num_classes= None, dtype = tf.int32, name=None, weights= None)
################
### ske LSTM ###
################
skeleton_input_size = np.shape(skeleton_train_data)[2] #
skeleton_timestep_size = np.shape(skeleton_train_data)[1] # timesteps
skeleton_hidden_size = 128 # hidden layer num of features
class_num = 20 #
skeleton_layer_num = 1

# LSTM Model
skeleton_X = tf.placeholder(tf.float32, [None, skeleton_timestep_size, skeleton_input_size])

skeleton_y = tf.placeholder(tf.float32, [None, class_num])
skeleton_keep_prob = tf.placeholder(tf.float32)



skeleton_mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(skeleton_hidden_size, skeleton_keep_prob, 'skeleton_lstm') for _ in range(skeleton_layer_num)])

skeleton_outputs, skeleton_state = tf.nn.dynamic_rnn(skeleton_mlstm_cell, inputs=skeleton_X, dtype=tf.float32)
# last one
skeleton_h_state = skeleton_outputs[:, -1, :]

skeleton_W = tf.Variable(tf.truncated_normal([skeleton_hidden_size, class_num], stddev=0.1), dtype=tf.float32)
skeleton_bias = tf.Variable(tf.constant(1.0, shape=[class_num]), dtype=tf.float32)

skeleton_y_pre = tf.nn.softmax(tf.matmul(skeleton_h_state, skeleton_W) + skeleton_bias)

skeleton_cross_entropy = tf.reduce_mean(tf.square(skeleton_y - skeleton_y_pre))

skeleton_result = tf.argmax(skeleton_y_pre, 1)
skeleton_target = tf.argmax(skeleton_y, 1)
skeleton_correct_prediction = tf.equal(tf.argmax(skeleton_y_pre, 1), tf.argmax(skeleton_y, 1))
skeleton_accuracy = tf.reduce_mean(tf.cast(skeleton_correct_prediction, "float"))
skeleton_confusion_matrix = tf.contrib.metrics.confusion_matrix(skeleton_result, skeleton_target, num_classes= None, dtype = tf.int32, name=None, weights= None)


##########################################################################
#################
### rgb graph ###
#################
rgb_y_pre_1 = tf.expand_dims(rgb_y_pre, -1)
rgb_y_pre_2 = tf.expand_dims(rgb_y_pre, 1)
rgb_rgb_matrix = tf.matmul(rgb_y_pre_1, rgb_y_pre_2)
#################
### dep graph ###
#################
depth_y_pre_1 = tf.expand_dims(depth_y_pre, -1)
depth_y_pre_2 = tf.expand_dims(depth_y_pre, 1)
depth_depth_matrix = tf.matmul(depth_y_pre_1, depth_y_pre_2)
#################
### ske graph ###
#################
skeleton_y_pre_1 = tf.expand_dims(skeleton_y_pre, -1)
skeleton_y_pre_2 = tf.expand_dims(skeleton_y_pre, 1)
skeleton_skeleton_matrix = tf.matmul(skeleton_y_pre_1, skeleton_y_pre_2)
#####################
### mixed graph ###
#####################

# RGB and SKELETON
rgb_skeleton_matrix = tf.matmul(rgb_y_pre_1, skeleton_y_pre_2)
# RGB and DEPTH
rgb_depth_matrix = tf.matmul(rgb_y_pre_1, depth_y_pre_2)
# DEPTH and SKELETON
depth_skeleton_matrix = tf.matmul(depth_y_pre_1, skeleton_y_pre_2)
####################
### stack matrix ###
####################
rgb_rgb_matrix_expand = tf.expand_dims(rgb_rgb_matrix, -1)
depth_depth_matrix_expand = tf.expand_dims(depth_depth_matrix, -1)
skeleton_skeleton_matrix_expand = tf.expand_dims(skeleton_skeleton_matrix, -1)

rgb_skeleton_matrix_expand = tf.expand_dims(rgb_skeleton_matrix, -1)
rgb_depth_matrix_expand = tf.expand_dims(rgb_depth_matrix, -1)
depth_skeleton_matrix_expand = tf.expand_dims(depth_skeleton_matrix, -1)

stack_matrix = tf.concat([rgb_rgb_matrix_expand, 
    skeleton_skeleton_matrix_expand, 
    depth_depth_matrix_expand,
    rgb_skeleton_matrix_expand,
    rgb_depth_matrix_expand,
    depth_skeleton_matrix_expand], -1)

######################
### pixel wise CNN ###
######################

num_filter = 64
size_kernel = [1,1]
conv1 = tf.layers.conv2d(
    inputs = stack_matrix,
    filters = num_filter,
    kernel_size = size_kernel,
    strides = 1,
    padding = 'valid',
    activation = tf.nn.relu
)
flat = tf.reshape(conv1, [-1, class_num*class_num*num_filter])

########################
### final classifier ###
########################
classifier_y = tf.placeholder(tf.float32, [None, class_num])

classifier_W = tf.Variable(tf.truncated_normal([class_num*class_num*num_filter, class_num], stddev=0.1), dtype=tf.float32)
classifier_bias = tf.Variable(tf.constant(1.0, shape=[class_num]), dtype=tf.float32)

classifier_y_pre = tf.nn.softmax(tf.matmul(flat, classifier_W) + classifier_bias)
classifier_cross_entropy = tf.reduce_mean(tf.square(classifier_y - classifier_y_pre))

classifier_result = tf.argmax(classifier_y_pre, 1)
classifier_target = tf.argmax(classifier_y, 1) # 
classifier_correct_prediction = tf.equal(tf.argmax(classifier_y_pre, 1), tf.argmax(classifier_y, 1))
classifier_accuracy = tf.reduce_mean(tf.cast(classifier_correct_prediction, "float"))
classifier_confusion_matrix = tf.contrib.metrics.confusion_matrix(classifier_result, classifier_target, num_classes= None, dtype = tf.int32, name=None, weights= None)
##########################
### finish build model ###
##########################
alpha = 0.1
beta = 0.4
gamma = 0.5
lbd = 0.5

batch_size = 128

lr = 0.0001

final_cross_entropy = alpha*rgb_cross_entropy + beta*depth_cross_entropy + gamma*skeleton_cross_entropy + lbd*classifier_cross_entropy


optimizer = tf.train.AdamOptimizer(learning_rate=lr)


train_all = optimizer.minimize(final_cross_entropy)
train_rgb = optimizer.minimize(rgb_cross_entropy)
train_depth = optimizer.minimize(depth_cross_entropy)
train_skeleton = optimizer.minimize(skeleton_cross_entropy)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(500):
    print(i)
    for _ in range(100):
        rgb_train_x, depth_train_x, skeleton_train_x, train_y = next_batch_threeview(batch_size, rgb_train_data, depth_train_data, skeleton_train_data, rgb_train_label)
        # update weights
        for _ in range(10):
        
            sess.run(train_skeleton, feed_dict={skeleton_X: skeleton_train_x, skeleton_y: train_y, skeleton_keep_prob:1})
        for _ in range(2):
            sess.run(train_rgb, feed_dict={rgb_X: rgb_train_x, rgb_y: train_y, rgb_keep_prob:1})
            sess.run(train_depth, feed_dict={depth_X: depth_train_x, depth_y: train_y, depth_keep_prob:1})
        for _ in range(2):
            sess.run(train_all, feed_dict={rgb_X: rgb_train_x, depth_X: depth_train_x, skeleton_X: skeleton_train_x, rgb_y: train_y, depth_y: train_y, skeleton_y: train_y, rgb_keep_prob: 1, depth_keep_prob: 1, skeleton_keep_prob: 1, classifier_y: train_y})
        # sess.run()

    # calculate loss
    rgb_loss = sess.run(rgb_cross_entropy, feed_dict={rgb_X: rgb_train_x, rgb_y: train_y, rgb_keep_prob: 1})
    depth_loss = sess.run(depth_cross_entropy, feed_dict={depth_X: depth_train_x, depth_y: train_y, depth_keep_prob: 1})
    skeleton_loss = sess.run(skeleton_cross_entropy, feed_dict = {skeleton_X: skeleton_train_x, skeleton_y: train_y, skeleton_keep_prob: 1})
    classifier_loss = sess.run(classifier_cross_entropy, feed_dict = {rgb_X: rgb_train_x, depth_X: depth_train_x, skeleton_X: skeleton_train_x, classifier_y: train_y, rgb_keep_prob:1, depth_keep_prob: 1, skeleton_keep_prob:1})
    # calculate accuracy
    rgb_train_accuracy = sess.run(rgb_accuracy, feed_dict={rgb_X: rgb_train_x, rgb_y: train_y, rgb_keep_prob: 1})
    rgb_test_accuracy = sess.run(rgb_accuracy, feed_dict={rgb_X: rgb_test_data, rgb_y: rgb_test_label, rgb_keep_prob: 1})

    depth_train_accuracy = sess.run(depth_accuracy, feed_dict={depth_X: depth_train_x, depth_y: train_y, depth_keep_prob: 1})
    depth_test_accuracy = sess.run(depth_accuracy, feed_dict={depth_X: depth_test_data, depth_y: depth_test_label, depth_keep_prob: 1})

    skeleton_train_accuracy = sess.run(skeleton_accuracy, feed_dict={skeleton_X: skeleton_train_x, skeleton_y: train_y, skeleton_keep_prob: 1})
    skeleton_test_accuracy = sess.run(skeleton_accuracy, feed_dict= {skeleton_X: skeleton_test_data, skeleton_y: skeleton_test_label, skeleton_keep_prob: 1})

    classifier_train_accuracy = sess.run(classifier_accuracy, feed_dict={rgb_X: rgb_train_x, depth_X: depth_train_x, skeleton_X: skeleton_train_x, classifier_y: train_y, rgb_keep_prob:1, depth_keep_prob:1, skeleton_keep_prob:1})
    classifier_test_accuracy = sess.run(classifier_accuracy, feed_dict ={rgb_X: rgb_test_data, depth_X: depth_test_data, skeleton_X: skeleton_test_data, classifier_y: rgb_test_label, rgb_keep_prob:1, depth_keep_prob:1, skeleton_keep_prob:1})# use rgb label as final label
    
    print('iter: ', i*100, '\n',
          
          'train acc: ', rgb_train_accuracy, depth_train_accuracy, skeleton_train_accuracy, classifier_train_accuracy, '\n',
          'test acc: ', rgb_test_accuracy, depth_test_accuracy, skeleton_test_accuracy, classifier_test_accuracy, '\n',
          'loss:', rgb_loss, depth_loss, skeleton_loss, classifier_loss, '\n'
          
          )
