#====================================================================
#ASSIGMENT 2 UDACITY
#FIRST MULTI-LAYER NEURAL NETWORK
#WITH TENSORBOARD VISUALIZATION


#These are all the modules we'll be using later. Make sure you can import them
#before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import time
import os


#<<<<<<<<<<<<<<<<<<<<<<<<<loading data (begining)

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

#<<<<<<<<<<<<<<<<<<<<<<<<<loading data (end)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


#function that performs Xavier initialization   
def weights_Xavier_init(V_shape, name):
    return  tf.get_variable(name,\
    shape=V_shape,\
    initializer=tf.contrib.layers.xavier_initializer()) 

#function that performs the bias initialization  
def bias_init(V_shape,V_name):
    return tf.Variable(tf.zeros(V_shape), name=V_name)

#L2 regularization
def L2(beta,layer):
    return tf_beta*tf.nn.l2_loss(layer['weights'])
    
#linear function of layer layer using data as input 
def linear_activation(data,layer):
    return tf.add(tf.matmul(data, layer['weights'], name = 'multiply'),\
      layer['biases'],name ='add')  
 
#initialization of a otimizer (SGD) with exponential decay
def sgd_train(error, starter_learning_rate,steps_for_decay,decay_rate):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(\
                starter_learning_rate,\
                global_step,\
                steps_for_decay,\
                decay_rate, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(error, global_step=global_step)


#the model of the network: 4 layers,
#activators: sigmoid --> relu --> sigmoid
def network(\
            data,\
            V_input_layer,\
            V_hidden_layer_1,\
            V_hidden_layer_2,\
            V_hidden_layer_3):
    with tf.name_scope('Input_Layer'):
        l1 = tf.sigmoid(linear_activation(data, V_input_layer))
    with tf.name_scope('Hidden_Layer_1'):
        l2 = tf.nn.relu(linear_activation(l1,V_hidden_layer_1)) 
    with tf.name_scope('Hidden_Layer_2'):
        l3 = tf.sigmoid(linear_activation(l2,V_hidden_layer_2))
    with tf.name_scope('Output_Layer'):
        logits = linear_activation(l3,V_hidden_layer_3) 
        return logits

    

batch_size = 128
hidden_nodes_1 = 60
hidden_nodes_2 = 40
hidden_nodes_3 = 20


graph = tf.Graph()
with graph.as_default():
    
    #placeholders for the SGD
    tf_train_dataset = tf.placeholder(tf.float32,\
        shape=(batch_size, image_size* image_size), name ='X')
    
    tf_train_labels = tf.placeholder(tf.float32, \
        shape=(batch_size, num_labels),name ='Y')

    #constants: we use then to see accuracity of the network
    tf_valid_dataset = tf.constant(valid_dataset, name ='X_va')
    tf_test_dataset = tf.constant(test_dataset, name ='X_test')
    
    #constant for the L2 regularization 
    tf_beta = tf.constant(0.005)

    #input layer - Xavier initialization    
    tf_input_layer = {
        'weights': weights_Xavier_init([image_size * image_size, hidden_nodes_1],"weights1"),
        'biases':bias_init([hidden_nodes_1],'biases1')}

    #hidden layer 1  - Xavier initialization 
    tf_hidden_layer_1 = {
        'weights':weights_Xavier_init([hidden_nodes_1, hidden_nodes_2],"weights2"),
        'biases': bias_init([hidden_nodes_2],'biases2')}
    
    #hidden layer 2 - normal initialization 
    tf_hidden_layer_2 = {
        'weights':tf.Variable(tf.truncated_normal([hidden_nodes_2,hidden_nodes_3]), name="weights3"),
        'biases':  bias_init([hidden_nodes_3],'biases3')}


    #hidden layer 3 - normal initialization 
    tf_hidden_layer_3 = {
        'weights':tf.Variable(tf.truncated_normal([hidden_nodes_3,num_labels]),name="weights4"),
        'biases':bias_init([num_labels],'biases4')}
    
    #histogram summaries for weights
    tf.histogram_summary('weights1_summ',tf_input_layer['weights'])
    tf.histogram_summary('weights2_summ',tf_hidden_layer_1['weights'])
    tf.histogram_summary('weights3_summ',tf_hidden_layer_2['weights'])
    tf.histogram_summary('weights4_summ',tf_hidden_layer_3['weights'])
        
    #the NN with the train dataset
    train_network = network(\
                          tf_train_dataset,\
                          tf_input_layer,\
                          tf_hidden_layer_1,\
                          tf_hidden_layer_2,\
                          tf_hidden_layer_3)

    #the NN with the valid dataset
    valid_network = network(\
                      tf_valid_dataset,\
                      tf_input_layer,\
                      tf_hidden_layer_1,\
                      tf_hidden_layer_2,\
                      tf_hidden_layer_3)
    
    #the NN with the test dataset
    test_network = network(\
                      tf_test_dataset,\
                      tf_input_layer,\
                      tf_hidden_layer_1,\
                      tf_hidden_layer_2,\
                      tf_hidden_layer_3)
   
    
    #loss function that measures the distance between the network predictions
    #and the target labels
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
            train_network, tf_train_labels))\
        #+ L2(tf_beta,tf_input_layer)\
        #+ L2(tf_beta,tf_hidden_layer_1)\
        #+ L2(tf_beta,tf_hidden_layer_2)\
        #+ L2(tf_beta,tf_hidden_layer_3)
        tf.scalar_summary(loss.op.name,loss) #write loss to log
        
    #Optimizer.
    with tf.name_scope('training'):
        optimizer = sgd_train(loss, 0.9,100,0.96)
        
    #applying the softmax function on the 
    #network with the train, valid,
    #and test datasets, respectively.       
    train_prediction = tf.nn.softmax(train_network, name='train_network')
    valid_prediction = tf.nn.softmax(valid_network, name='valid_network')
    test_prediction = tf.nn.softmax(test_network, name='test_network') 
    
    #Minibatch accuracy
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(tf_train_labels, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_pred,'float'))
        tf.scalar_summary(acc_op.op.name,acc_op) #write acc to log
    
    
#direction for the writer to log
log_basedir = 'logs'
run_label = time.strftime('%d-%m-%Y_%H-%M-%S') #e.g. 12-11-2016_18-20-45
log_path = os.path.join(log_basedir,run_label)

#number of iterations
num_steps = 3001


#begining tf session
with tf.Session(graph=graph) as session:
  summary_writer = tf.train.SummaryWriter(log_path, session.graph) 
  all_summaries = tf.merge_all_summaries()      
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
   
    # Pick an offset within the training data, which has been randomized.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    # Prepare a dictionary telling the session where to feed the minibatch.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    
    #we are going to run the session and count the duration of the running
    start_time = time.time()
    _, l, predictions, acc, summary = session.run(
      [optimizer, loss, train_prediction, acc_op, all_summaries], feed_dict=feed_dict)
    duration = time.time() - start_time
    
    #writing the log
    summary_writer.add_summary(summary,step)
    summary_writer.flush()
    
    #Printing an overwiew
    if (step % 500 == 0):
        print("Minibatch loss at step %d: %f" % (step, l))
        print("Minibatch accuracy: %.2f%%" % (acc*100))
        print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
        print('Duration: %.3f sec' % duration)
        
  #after the loop compair our model with the test dataset      
  print("Test accuracy: %.1f%%" % \
    accuracy(test_prediction.eval(), test_labels))

print(' ')
print(log_path)
#!tensorboard --logdir=!!!copy log_path here!!!