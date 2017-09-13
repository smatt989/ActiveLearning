from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
import numpy as np
import sys

######SEE BOTTOM FOR SCRIPT RUN

######Pulls reads in MNSIT data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

######Some options for different label functions
######Used for different classification tasks

###Here is the "3 or not-3" labeling function
def find_3s(old_label):
    if (old_label[3] == 1):
        return [1, 0]
    else:
        return [0, 1]

###Standard MNIST 10 digit classification labeling function
def keep_old(old_label):
    return old_label

###Separates 0-4 and 5-9 into two classes
def split_labels(old_label):
    if (np.sum(old_label[:5]) == 1):
        return [1, 0]
    else:
        return [0, 1]

###Default is "3 or not-3" labeling function
new_labels = find_3s

###assign the labeling function based on 5th parameter passed to the script
###choices are 'old' (0-9 classification), '3s' (3 or not-3 classifcation),
###or 'split' (0-4 or 5-9 classification)
if (len(sys.argv) >= 6):
    if (sys.argv[5] == "old"):
        new_labels = keep_old
    elif (sys.argv[5] == "split"):
        new_labels = split_labels
    else:
        new_labels = find_3s

######Ouptut layer size determined by labeling function
output_layer_size = len(new_labels(mnist.train.labels[0]))

######Here is the neural net model described in Tensor Flow MNIST example
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x  = tf.placeholder(tf.float32, [None, 784], name='x')
x_image = tf.reshape(x, [-1, 28, 28, 1])

y_ = tf.placeholder(tf.float32, [None, output_layer_size],  name='y_')

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, output_layer_size])
b_fc2 = bias_variable([output_layer_size])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

######A function that randomly generates a value to be used when selecting examples
######for mini-batch training when not using active learning
def random_values(l):
    return [random.randint(0,9) for b in l]

######Ranks mnist training images according to a rank function (random for normal, 
######model evaluation for active learning)
def choose_examples(datas = mnist.train.images, batch_size = 50, rank_function = random_values, chosen = []):

    look_size = batch_size * 100
    
    training_lookup_index = range(0, len(mnist.train.images))

    #do not look at examples that have already been seen
    raw_remain = list(set(training_lookup_index) - set(chosen))
    
    #shuffle the unseen examples to randomize order
    random.shuffle(raw_remain)
    
    looking_in = []
    
    #look at the first batch_size * 100 unseen exmaples (shuffled)
    if (look_size >= len(raw_remain)):
        looking_in = raw_remain
    else:
        looking_in = raw_remain[:look_size]
        
    remain_data = [datas[k] for k in looking_in]
    
    #rank the examples according to the rank function
    ranks = rank_function(remain_data)
    scores = np.column_stack((looking_in, ranks))
    to_return = []

    scores = np.array(scores)
        
    selected = []
        
    #select examples based on their scores, enough to fill a batch
    if len(scores) >= batch_size:  
        sort = scores[np.argsort(scores[:,1])]
        selected = sort[:batch_size]
    else:
        selected = scores

    #return the index value for each chosen example
    return [int(s[0]) for s in selected]

######Big mess of a function that does a lot of things
def run_batch(runs, size, max_steps, active, extra_sample, print_every):
    #collects information about each run
    batch_log = []

    #converts mnist test labels into new labels given label function
    test_labels = [new_labels(b) for b in mnist.test.labels]

    for i in range(runs):
        #previously trained on examples
        chosen = []

        #order that examples were selected to be trained on
        ordered = []

        #collects information about a given run
        run_log = []

        with tf.Session() as sess:
            #initializes the model
            sess.run(tf.initialize_all_variables())

            #rank function for active learning... should not be in here
            def rank_function(examples):
                return [abs(np.max(s) - np.min(s)) for s in sess.run(y, feed_dict={x: examples, keep_prob: 1.0})]

            #rank function defaults to random, if active parameter passed
            #then will be the active learning rank function
            ranker = random_values
            
            if active:
                ranker = rank_function
            
            #iterate for each step (mini-batch)
            for step in range(max_steps):
                #determines which train examples to look at in this mini-batch
                next_batch = choose_examples(mnist.train.images, size, ranker, chosen)

                #if re-sampling turned on, will double size of the mini-batch
                #by randomly sampling from previously trained on exmaples
                to_train = next_batch
                if extra_sample and len(chosen) > 0:
                    random.shuffle(chosen)
                    to_train = chosen[:size] + next_batch

                chosen = chosen + next_batch
                ordered = ordered + next_batch

                #if all examples have been trained on, will start over (new epoch kind of)
                if(len(chosen) == len(mnist.train.labels)):
                    chosen = []

                #grabs images and labels based on mini-batch
                batch_xs = [mnist.train.images[s] for s in to_train]
                batch_ys = [mnist.train.labels[s] for s in to_train]

                #counts how many positive examples were in a mini-batch
                positive_examples = 0
                changed_ys = [new_labels(ys) for ys in batch_ys]
                for ys in changed_ys:
                    positive_examples = positive_examples + ys[0]

                #creates test results every print_every mini-batches
                #passed in as a parameter
                if (step % print_every) == 0:
                    rr = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: test_labels, keep_prob: 1.0})
                    acc = rr[0]
                    ce = rr[1]
                    print(acc)
                    run_log.append([step, acc, ce, float(positive_examples)/len(to_train)])
                
                #trains the model using the mini-batch
                sess.run(train_step, feed_dict={x: batch_xs, y_: changed_ys, keep_prob: 0.5})
            
            #after all the mini-batch training, run against test set and generate results
            final_rr = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: test_labels, keep_prob: 1.0})
            final = final_rr[0]
            final_ce = final_rr[1]
            print(max_steps, final)
            run_log.append([max_steps, final, final_ce, float(positive_examples)/len(to_train)])
            batch_log.append(run_log)
            print("done with run ", i)

            #start multi-epoch portion (kind of pasted on at the end)
            epoch_logs = []

            #evaluate every this many labels
            label_range = 250

            #this many epochs
            epochs = 20

            #mini-batch size
            epoch_mini_batch_size = 50

            #creates labeled data sets at label_range increments and runs multi-epoch model on that 
            for label_size in range(len(ordered)/label_range):
                labels_length = (label_size + 1) * label_range
                result = epoch_sample(ordered[0: labels_length], epoch_mini_batch_size, epochs, sess)
                epoch_logs.append([labels_length, epochs, result[0], result[1]])
            print("labels\tepoch\taccuracy\tcross entropy")
            for entry in epoch_logs:
                print(entry[0], "\t", entry[1], "\t", entry[2], "\t", entry[3])
    print("donezo")
    return batch_log

######trains one model on a subset of mnist data for a certian number of epochs
def epoch_sample(chosen, mini_batch_size, epochs, sess):
    #initializes the model (starts it over)
    sess.run(tf.initialize_all_variables())

    for epoch in range(epochs):
        print("starting epoch ", epoch)

        #shuffle the labeled dataset, and do mini-batch training
        random.shuffle(chosen)
        batches = len(chosen) / mini_batch_size
        for i in range(batches):
            end = (i + 1) * mini_batch_size
            if(end > len(chosen)):
                end = len(chosen) - 1
            training_data = [mnist.train.images[s] for s in chosen[i * mini_batch_size: end]]
            training_labels = [mnist.train.labels[s] for s in chosen[i * mini_batch_size: end]]
            sess.run(train_step, feed_dict={x: training_data, y_: [new_labels(b) for b in training_labels], keep_prob: 0.5})
    print("done epoch training")
    epoch_rr = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, y_: [new_labels(b) for b in mnist.test.labels], keep_prob: 1.0})
    epoch_acc = epoch_rr[0]
    epoch_ce = epoch_rr[1]
    print("labels: ", len(chosen))
    print("epochs: ", epochs)
    print("acc: ", epoch_acc)
    print("cross entropy: ", epoch_ce)
    return epoch_rr

def print_average_series(to_average, columns, column_names):
    labels = [int(a[0]) for a in to_average[0]]
    column_collection = []
    for column in columns:
        transformed = []
        for run in to_average:
            transformed.append([a[column] for a in run])
        column_collection.append(np.mean(np.transpose(transformed), axis=1))
    column_collection.insert(0, np.array(labels))
    column_names.insert(0, 'iteration')
    print(*column_names, sep='\t')
    for row in np.transpose(column_collection):
        print(*row, sep='\t')

def print_details(runs, batch_size, iterations, active, extra_sample):
    print("runs", runs)
    print("batch_size", batch_size)
    print("iterations", iterations)
    if(active):
        print("active")
    else:
        print("random")
    if(extra_sample):
        print("extra sample")
    else:
        print("standard sample")
    labels = [new_labels(b) for b in mnist.test.labels]
    print("positive, negative", np.sum(labels, axis=0))

######not my most creative name.  does a run, and then prints out the results
def make_a_good_test(runs, batch_size, iterations, active, extra_sample, print_every):
    print_details(runs, batch_size, iterations, active, extra_sample)
    results = run_batch(runs, batch_size, iterations, active, extra_sample, print_every)
    print_details(runs, batch_size, iterations, active, extra_sample)
    print_average_series(results, [1, 2, 3], ["accuracy", "cross entropy", "% positive examples"])

extra_sampling = False
print_every = 5

######re-sampling defaults to False, determined by 6th parameter
if(len(sys.argv) >= 7):
    extra_sampling = sys.argv[6] == 'True'

######test model interval defaults to 5, determined by 7th parameter
if(len(sys.argv) >= 8):
    print_every = int(sys.argv[7])


######runs a test with the following parameters:
###1st parameter: number of runs (almost always want just 1 or will be extremely long)
###2nd parameter: how bit each mini-batch should be
###3rd parameter: how many mini-batches in a run
###4th parameter: active learning turned on or not
###5th parameter: classification task ('old', '3s', 'split')
###6th parameter: re-samping turned on or not
###7th parameter: how often training is tested with test set
make_a_good_test(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == 'True', extra_sampling, print_every)
