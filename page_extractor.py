import numpy as np
import tensorflow as tf
from PIL import Image
from os import listdir, makedirs, rmdir, remove
from os.path import isdir, isfile, join, exists
from shutil import copyfile, rmtree

#settings

directory = "/home/ollie/work/fyp/page_trainer"
training_steps = 100
scaled_image_size = (32,32)


#extract training set

characters = [f for f in listdir(directory + "/training/") if isdir(join(directory + "/training/", f))]
characters = sorted(characters)
unique_characters = len(characters)

def label_generator(num):
    label = [0] * unique_characters
    label[num] = 1
    return label

def format_image(image):
    image = image.resize(scaled_image_size,resample = Image.BILINEAR)
    image = np.array(image)
    image = image[:, :, image.shape[2]-1]
    image = np.divide(image,255)
    image = image.flatten()
    return image

images, labels = [],[]

for char in characters:
    for name in [f for f in listdir(directory + "/training/" +char) if isfile(join(directory + "/training/"+char, f))]:
        with Image.open(directory + "/training/" + char + "/" + name) as imageFile:
            images.append(format_image(imageFile))
            labels.append(label_generator(characters.index(char)))

#training
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
																																																																																																													
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


#input layer
x  = tf.placeholder(tf.float32, [None, scaled_image_size[0] * scaled_image_size[1]], name='x')
y_ = tf.placeholder(tf.float32, [None, unique_characters],  name='y_')
x_image = tf.reshape(x, [-1, scaled_image_size[0], scaled_image_size[1], 1])

#convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, int(scaled_image_size[0]/4)*int(scaled_image_size[1]/4)*64])

W_fc1 = weight_variable([int(scaled_image_size[0]/4) * int(scaled_image_size[1]/4) * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, unique_characters])
b_fc2 = bias_variable([unique_characters])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

#evaluation function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#training steps
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print("Beginning training...")

for step in range(training_steps):
    sess.run(train_step, feed_dict={x: images, y_: labels, keep_prob: 0.5})
    print("    "+str(step)+"/"+str(training_steps), end="\r")

print("Training complete.")
print()

#analyse page

print("Analysing page...")

for removal in listdir(directory + "/classified/"):
    rmtree(directory + "/classified/" + removal)

for char in characters:
    makedirs(directory + "/classified/" + char)

makedirs(directory + "/classified/unknown")

unknown_characters = [f for f in listdir(directory + "/unclassified/") if isfile(join(directory + "/unclassified/", f))]
unknown_characters = unknown_characters[:1000]

def extract_value(prob_list):
    prob_list = prob_list[0]
    indexes = [i for i in range(len(prob_list)) if prob_list[i] > 0.999999]
    if len(indexes) == 0:
       return "unknown"
    return characters[indexes[0]]

counter = 0
for char in unknown_characters:
    with Image.open(directory + "/unclassified/" + char) as imageFile:
        image = format_image(imageFile)
    value = extract_value(sess.run(y,feed_dict={x: [image], keep_prob: 1.0}))
    copyfile(directory + "/unclassified/" + char,directory + "/classified/" + value+"/"+char)
    counter += 1
    print("    "+str(counter)+"/"+str(len(unknown_characters)), end="\r")
    
print("Analysis complete.")
print()


#remove empty dirs
for char in characters:
    try:
        rmdir(directory + "/classified/" + char) #throws exception for dirs with contents
    except:
        pass

try:
    rmdir(directory + "/classified/unknown")
except:
    pass

