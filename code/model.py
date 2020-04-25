from utils import randomMiniBatches
import tensorflow as tf

def createPlaceHolders(inputFeatures,numofClasses):
    X = tf.placeholder(dtype=tf.float32, shape=[inputFeatures, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[numofClasses, None])
    return X,Y

def initializeParameters():
    '''
    Hidden Layer 1:
        Here 25 is the number of neurons in Layer 1 where as 2352 is the number of Input Activations
        For Layer 1 Input Activations = Number of Features in Training Data
    '''
    W1 = tf.get_variable("W1", [25, 2352], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    '''
    Hidden Layer 2:
        Here 12 is the number of neurons in Layer 1 where as 25 is the number of Input Activations
        For Layer 2 Input Activations = Number of Neurons in Layer 1
    '''
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    '''
    Output Layer :
        Here 10 is the number of classes ,where as 12 is the number of Input Activations
        For Output Layer Input Activations = Number of Neurons in Last Hidden Layer
    '''
    W3 = tf.get_variable("W3", [10, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [10, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forwardPropogation(X,parameters):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X) , b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1) , b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2) , b3)

    return Z3

def computeCost(logits,labels):

    logits = tf.transpose(logits)
    labels = tf.transpose(labels)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

def model(trainData, trainLabels, testData, testLabels, learningRate = 0.0001,
          numEpochs = 1500, bacthSize = 32, printCost = True):
    '''
    Let's start building are model Step by Step
    '''
    # Make an empty cost list to append the cost of each epoch
    costs = []
    # initialize the seed for random miniBatch generation.
    seed = 0

    # Step 1 : Create the placeholders for your Input Data and Labels
    X, Y = createPlaceHolders(trainData.shape[0],trainLabels.shape[0])
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
            miniBatches = randomMiniBatches(trainData, trainLabels, bacthSize,seed)
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
