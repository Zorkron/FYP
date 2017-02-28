import numpy as np
import tensorflow as tf
from PIL import Image
from os import listdir
from os.path import isfile, join

#settings
np.set_printoptions(threshold=np.nan)
unique_characters = 10
number_of_images = 1
number_of_test_images = 1000
training_steps = 100
directory = "/home/ollie/work/fyp"
scaled_image_size = (32,32)


def extract_all(total):
    l = []
    for num in range(total):
        numbers = []
        names = [f for f in listdir(directory + "/digits/" + str(num)) if isfile(join(directory + "/digits/" + str(num), f))]
        total = 0
        for loc in names:
            if total < number_of_images + number_of_test_images:
                with Image.open(directory + "/digits/" + str(num) + "/" + loc) as imageFile:
                    image = imageFile
                    image = image.resize(scaled_image_size,resample = Image.BILINEAR)
                    image = np.array(image)
                    image = image[:, :, 3]
                    image = np.divide(image,255)
                    image = image.flatten()
                    numbers.append(image)
                total += 1
            else:
                break
        print("    "+str(num)+" complete")
        l.append(numbers)
    return l

#import images data
print("Loading raw images...")

all_images = extract_all(unique_characters)

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
for i in range(unique_characters):
    test_images.append(all_images[i][number_of_images:number_of_images+number_of_test_images])


#generate test lables

test_labels = []
for i in range(unique_characters):
    for j in range(number_of_test_images):
        test_labels.append(label_generator(i))


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
  #print(step, sess.run(accuracy, feed_dict={x: testImages, y_: testLabels, keep_prob: 1.0}))

print("Training complete.")
print()

#test

def analyse_result(result):
    results = []
    for i in range(unique_characters):
        results.append(result.count(i))
    return results

def analyse(number):
    numberImages = test_images[number]

    results = []

    for i in range(len(numberImages)):
        results.append(np.argmax(sess.run(y,feed_dict={x: [numberImages[i]], keep_prob: 1.0})))

    analysedResults = analyse_result(results)

    def percentageOfTotal(list,loc):
        return list[loc]/sum(list)*100

    print(str(number) + " - " +str(percentageOfTotal(analysedResults,number))+"%")
    print("--------------------")
    print("Number | Occurrences")
    current = 0
    for result in analysedResults:
        print(str(current)+"      | "+str(result))
        current += 1
    print("--------------------")
    print()

    return analysedResults[number]

totalCorrect = 0

for i in range(unique_characters):
    totalCorrect += analyse(i)

print("Overall accuracy: "+str(totalCorrect*100/(number_of_test_images*unique_characters))+"%")
