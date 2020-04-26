# Import all the necessary file and packages
from utils import one_hot_matrix
from loadDataset import loadDataset
from visualization import  show_images
import numpy as np
import yaml

if __name__ == '__main__':
    # read the configuration file
    with open(r'config.yaml') as file:
        cfg = yaml.load(file)

    # load the the training and testing data from the corresponding directories
    dataset = loadDataset(cfg)
    # Select few random Images for display purpose
    randomImages = dataset.trainData[np.random.randint(dataset.trainData.shape[0], size=25)]
    # display Images
    show_images(randomImages)

    # Display the shapes for Train , Test Data and Labels
    print("training data before Flatten and One-Hot Encoding")
    print(dataset.trainData.shape, dataset.trainLabel.shape)
    print("testing data before Flatten and One-Hot Encoding")
    print(dataset.testData.shape, dataset.testLabel.shape)

    if cfg['network'] == 0:
        '''
        Since we are going to use a Multi-Layer Perceptron Model :
            Flatten the Training and Testing Data.
            After Flattening Normalize the Data between 0-1
        The problem is a Multi-Class Classification :
            One Hot encode your Labels for both Train and Test
        '''
        # Transform the Input Data
        trainData = dataset.trainData.reshape(dataset.trainData.shape[0], -1).T
        testData = dataset.testData.reshape(dataset.testData.shape[0], -1).T

        trainLabels = dataset.trainLabel
        # 10 is for number of classes
        trainLabels = one_hot_matrix(trainLabels, 10).T
        testLabels = dataset.testLabel
        # 10 is for number of classes7
        testLabels = one_hot_matrix(testLabels, 10).T
    else :
        '''
        Since we are going to use Convoltion Neural Network:
            We DO NOT require Flatten stage.
            But we still do want our Data to be Normalized
        '''
        trainData = dataset.trainData
        testData = dataset.testData

        trainLabels = dataset.trainLabel
        # 10 is for number of classes
        trainLabels = one_hot_matrix(trainLabels, 10)
        testLabels = dataset.testLabel
        # 10 is for number of classes7
        testLabels = one_hot_matrix(testLabels, 10)

    trainData = trainData / 255
    testData = testData / 255


    print("Training Data after Flatten and One-Hot Encoding")
    print(trainData.shape, trainLabels.shape)
    print("Testing Data after Flatten and One-Hot Encoding")
    print(testData.shape, testLabels.shape)

    # Get the model according to the network parameter
    if cfg['network'] == 0:
        # import the Multi-Layer Perceptron Model
        from modelMLP import model,predict
    else :
        # import the COnvolution Layer Model
        from modelCNN import model, predict

    parameters = model(trainData=trainData,trainLabels=trainLabels,testData=testData,testLabels=testLabels,
                       bacthSize=cfg['batchSize'])

    predictions = predict(parameters,testData,testLabels)

    print("True Labels : ",dataset.testLabel)
    print("Predicted Labels : ",predictions)