import numpy as np
import tensorflow as tf
from random import shuffle
from PIL import Image
from os import listdir
from os.path import isfile, join

#settings
np.set_printoptions(threshold=np.nan)
NumberOfImages = 1
NumberOfTestImages = 500


def extract(num):
    numbers = []
    names = [f for f in listdir("/home/ollie/Pictures/digits/" + str(num)) if isfile(join("/home/ollie/Pictures/digits/" + str(num), f))]
    shuffle(names)
    total = 0
    for loc in names:
        if total < NumberOfImages + NumberOfTestImages:
            with Image.open('/home/ollie/Pictures/digits/' + str(num) + '/' + loc) as imageFile:
                template = np.ones((35,35),dtype=np.int)
                a = np.array(imageFile)
                a[:] = [x / 255 for x in a]
                new = a[:, :, 0]
                template[0:new.shape[0],0:new.shape[1]] = new
                template = template.flatten()
                template[template == 0] = 255
                template[template == 1] = 0
                template[template == 255] = 1
                numbers.append(template)
            total += 1
        else:
            break
    return numbers

#import images data
print("Loading raw images...")

zeros = extract(0)
ones = extract(1)
twos = extract(2)
threes = extract(3)
fours = extract(4)
fives = extract(5)
sixes = extract(6)
sevens = extract(7)
eights = extract(8)
nines = extract(9)

print("Import complete.")

#generate images
print("Generating training images...")

images = []
images.extend(zeros[:NumberOfImages])
images.extend(ones[:NumberOfImages])
images.extend(twos[:NumberOfImages])
images.extend(threes[:NumberOfImages])
images.extend(fours[:NumberOfImages])
images.extend(fives[:NumberOfImages])
images.extend(sixes[:NumberOfImages])
images.extend(sevens[:NumberOfImages])
images.extend(eights[:NumberOfImages])
images.extend(nines[:NumberOfImages])

#generate labels
print("Generating training labels...")

labels = []
for i in range(NumberOfImages):
    labels.append([1,0,0,0,0,0,0,0,0,0])
for i in range(NumberOfImages):
    labels.append([0,1,0,0,0,0,0,0,0,0])
for i in range(NumberOfImages):
    labels.append([0,0,1,0,0,0,0,0,0,0])
for i in range(NumberOfImages):
    labels.append([0,0,0,1,0,0,0,0,0,0])
for i in range(NumberOfImages):
    labels.append([0,0,0,0,1,0,0,0,0,0])
for i in range(NumberOfImages):
    labels.append([0,0,0,0,0,1,0,0,0,0])
for i in range(NumberOfImages):
    labels.append([0,0,0,0,0,0,1,0,0,0])
for i in range(NumberOfImages):
    labels.append([0,0,0,0,0,0,0,1,0,0])
for i in range(NumberOfImages):
    labels.append([0,0,0,0,0,0,0,0,1,0])
for i in range(NumberOfImages):
    labels.append([0,0,0,0,0,0,0,0,0,1])

#generate test images
print("Generating test images...")

testZeroes = zeros[NumberOfImages:NumberOfImages+NumberOfTestImages]
testOnes = ones[NumberOfImages:NumberOfImages+NumberOfTestImages]
testTwos = twos[NumberOfImages:NumberOfImages+NumberOfTestImages]
testThrees = threes[NumberOfImages:NumberOfImages+NumberOfTestImages]
testFours = fours[NumberOfImages:NumberOfImages+NumberOfTestImages]
testFives = fives[NumberOfImages:NumberOfImages+NumberOfTestImages]
testSixes = sixes[NumberOfImages:NumberOfImages+NumberOfTestImages]
testSevens = sevens[NumberOfImages:NumberOfImages+NumberOfTestImages]
testEights = eights[NumberOfImages:NumberOfImages+NumberOfTestImages]
testNines = nines[NumberOfImages:NumberOfImages+NumberOfTestImages]

#training
print("Setting up training...")
x = tf.placeholder(tf.float32, [None, 1225])
W = tf.Variable(tf.zeros([1225, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print("Training...")
sess.run(train_step, feed_dict={x: images, y_: labels})

#test

def largestMember(l):
    return np.argmax(l)

def analyseResult(result):
    results = []
    for i in range(10):
        results.append(result.count(i))
    return results

def analyse(number):
    numberImages = []

    if number == 0:
        numberImages = testZeroes
    if number == 1:
        numberImages = testOnes
    if number == 2:
        numberImages = testTwos
    if number == 3:
        numberImages = testThrees
    if number == 4:
        numberImages = testFours
    if number == 5:
        numberImages = testFives
    if number == 6:
        numberImages = testSixes
    if number == 7:
        numberImages = testSevens
    if number == 8:
        numberImages = testEights
    if number == 9:
        numberImages = testNines

    results = []

    for i in range(len(numberImages)):
        results.append(largestMember(sess.run(y,feed_dict={x: [numberImages[i]]})))

    analysedResults = analyseResult(results)

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

for i in range(10):
    totalCorrect += analyse(i)

print("Overall accuracy: "+str(totalCorrect*100/(NumberOfTestImages*10))+"%")