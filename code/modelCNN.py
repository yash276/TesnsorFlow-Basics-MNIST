from matplotlib import pyplot as plt
from utils import randomMiniBatchesCNN
import tensorflow as tf
import numpy as np

def createPlaceHolders(imageHeight,imageWidth,channels,numofClasses):
    X = tf.placeholder(dtype=tf.float32, shape=[None,imageHeight,imageWidth,channels])
    Y = tf.placeholder(dtype=tf.float32, shape=[None,numofClasses])
    return X,Y

def initializeParameters():
    '''
    We are going to Define 2 Convolution Layers
    Here [5,5,3,8] is [filterHeight , filterWidth , filterChannels ,No. Of filters] for Layer 1
    Same is the case with layer 2.
    '''
    W1 = tf.get_variable("W1", [5,5,3,8], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable("b1", [3,3,8,16], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    parameters = {"W1": W1,
                  "W2": W2}

    return parameters

def forwardPropogation(X,parameters):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    F = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 10 neurons in output layer because we have 10 classes to pre77dict. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(F, 10, activation_fn=None)

    return Z3

def computeCost(logits,labels):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

def model(trainData, trainLabels, testData, testLabels, learningRate = 0.0001,
          numEpochs = 150, bacthSize = 32, printCost = True):
    '''
    Let's start building are model Step by Step
    '''
    # Make an empty cost list to append the cost of each epoch
    costs = []
    # initialize the seed for random miniBatch generation.
    seed = 0

    # Step 1 : Create the placeholders for your Input Data and Labels
    X, Y = createPlaceHolders(trainData.shape[1],trainData.shape[2],trainData.shape[3],trainLabels.shape[1])
    # Step 2 : We will a network that has 2 Hidden Layers and 1 Output Layer
    # Create the Weight Matrix and Bias Vector for each layer
    parameters = initializeParameters()
    # Step 3 : Let's define our Forward Propogation Algortihm
    logits = forwardPropogation(X, parameters)
    # Step 4 : Define our Loss Functions for Multi-Class Classification
    cost = computeCost(logits, Y)
    # Step 5 : Define BackPropogation to minimize Cost Function
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

    # Time to run the Graph and Train the Model
    # Initialize all the variables in the Grapth
    init = tf.global_variables_initializer()

    # Start a new tensorFlow Session
    with tf.Session() as sess:
        print("Training the CNN model")
        # run the initialization
        sess.run(init)
        # The training loop
        for epoch in range(numEpochs):
            '''
            Since we are going to use Mini-Batch Gradient Optimizer
                Generate miniBatches from the Training Data.
            '''
            # For a new epoch set cost to 0
            epochCost = 0
            seed += 1
            miniBatches = randomMiniBatchesCNN(trainData, trainLabels, bacthSize,seed)
            # iterate over all the minibatch and update the parameters
            for miniBatch in miniBatches:
                (miniBatchX , miniBatchY) = miniBatch
                _, minibatchCost = sess.run(fetches=[optimizer, cost],
                                             feed_dict={X: miniBatchX, Y: miniBatchY})

                epochCost += minibatchCost/bacthSize
            # Print the cost every epoch
            if printCost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epochCost))
            if printCost == True and epoch % 5 == 0:
                costs.append(epochCost)
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learningRate))
        plt.savefig("costCNN.png")
        plt.show()
        # Let's save the learned parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: trainData, Y: trainLabels}))
        print("Test Accuracy:", accuracy.eval({X: testData, Y: testLabels}))

        return parameters

def predict(parameters,testData,testLabels):

    # convert them to tensors
    W1 = tf.convert_to_tensor(parameters['W1'])
    W2 = tf.convert_to_tensor(parameters['W2'])
    # create the parameter dictionary for forward pass
    params = {"W1" : W1,
              "W2" : W2}
    # create the input placeHolder
    _ , imageHeight, imageWidth, channels = testData.shape
    X = tf.placeholder(dtype=tf.float32, shape=[None, imageHeight, imageWidth, channels])
    # make the forward pass node
    Z3 = forwardPropogation(X,params)
    output = tf.argmax(Z3,1)
    # run the Session and get the predictions
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        predictions = sess.run(output, feed_dict={X:testData})
    return predictions