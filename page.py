import numpy as np
import tensorflow as tf
from PIL import Image
from os import listdir
from os.path import isfile, join

#settings
np.set_printoptions(threshold=np.nan)
unique_characters = 8
number_of_images = 10
number_of_test_images = 1
training_steps = 100
directory = "/home/ollie/work/fyp"
scaled_image_size = (32,32)

if number_of_images + number_of_test_images > 1945: #1945 0s in dataset
    print("number_of_images + number_of_test_images > 1945")
    quit()

def format_image(image):
    image = image.resize(scaled_image_size,resample = Image.BILINEAR)
    image = np.array(image)
    image = image[:, :, image.shape[2]-1]
    image = np.divide(image,255)
    image = image.flatten()
    return image

def extract_all(total):
    l = []
    for num in range(total):
        numbers = []
        names = [f for f in listdir(directory + "/digits/" + str(num)) if isfile(join(directory + "/digits/" + str(num), f))]
        total = 0
        for loc in names:
            if total < number_of_images + number_of_test_images:
                with Image.open(directory + "/digits/" + str(num) + "/" + loc) as imageFile:
                    image = format_image(imageFile)
                    numbers.append(image)
                total += 1
            else:
                break
        print("    "+str(num)+" complete", end="\r")
        l.append(numbers)
    return l

#import images data
print("Loading raw images...")

all_images = extract_all(10)

print("Import complete.")

#generate images
print("Generating training images...")

images = []

for i in range(unique_characters):
    images.extend(all_images[i][:number_of_images])

#generate labels
print("Generating training labels...")

def label_generator(num):
    label = [0] * unique_characters
    label[num] = 1
    return label

labels = []
for i in range(unique_characters):
    for j in range(number_of_images):
        labels.append(label_generator(i))

#generate test images
print("Generating test images...")

test_images = []
for i in range(10):
    test_images.append(all_images[i][number_of_images:number_of_images+number_of_test_images])


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
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Input layer
x  = tf.placeholder(tf.float32, [None, scaled_image_size[0] * scaled_image_size[1]], name='x')
y_ = tf.placeholder(tf.float32, [None, unique_characters],  name='y_')
x_image = tf.reshape(x, [-1, scaled_image_size[0], scaled_image_size[1], 1])

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
h_pool2_flat = tf.reshape(h_pool2, [-1, int(scaled_image_size[0]/4)*int(scaled_image_size[1]/4)*64])

W_fc1 = weight_variable([int(scaled_image_size[0]/4) * int(scaled_image_size[1]/4) * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, unique_characters])
b_fc2 = bias_variable([unique_characters])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print("Beginning training...")

for step in range(training_steps):
  sess.run(train_step, feed_dict={x: images, y_: labels, keep_prob: 0.5})
  print("    "+str(step)+"/"+str(training_steps), end="\r")

print("Training complete.")
print()

#test

def extract_value(prob_list):
    prob_list = prob_list[0]
    indexes = [i for i in range(len(prob_list)) if prob_list[i] > 0.999999]
    if len(indexes) == 0:
       return -1
    return indexes[0]
    
with Image.open("/home/ollie/work/fyp/digits/unknown/c03983_x1794_y2311.tif") as imageFile:
    im = imageFile
    testImage1 = format_image(im)

with Image.open("/home/ollie/work/fyp/digits/unknown/c06669_x1155_y3430.tif") as imageFile:
    im = imageFile
    testImage2 = format_image(im)

with Image.open("/home/ollie/work/fyp/digits/unknown/c03921_x0960_y2233.tif") as imageFile:
    im = imageFile
    testImage3 = format_image(im)

with Image.open("/home/ollie/work/fyp/digits/unknown/c10631_x0682_y5156.tif") as imageFile:
    im = imageFile
    testImage4 = format_image(im)

print(extract_value(sess.run(y,feed_dict={x: [test_images[0][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[1][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[2][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[3][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[4][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[5][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[6][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[7][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[8][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [test_images[9][0]], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [testImage1], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [testImage2], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [testImage3], keep_prob: 1.0})))
print(extract_value(sess.run(y,feed_dict={x: [testImage4], keep_prob: 1.0})))

