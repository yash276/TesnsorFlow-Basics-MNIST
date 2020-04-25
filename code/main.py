# Import all the necessary file and packages
from model import model
from utils import one_hot_matrix
from loadDataset import loadDataset
import yaml

if __name__ == '__main__':
    # read the configuration file
    with open(r'config.yaml') as file:
        cfg = yaml.load(file)

    # load the the training and testing data from the corresponding directories
    dataset = loadDataset(cfg)
    # Display the shapes for Train , Test Data and Labels
    print("training data before Flatten and One-Hot Encoding")
    print(dataset.trainData.shape, dataset.trainLabel.shape)
    print("testing data before Flatten and One-Hot Encoding")
    print(dataset.testData.shape, dataset.testLabel.shape)
    '''
    Since we are going to use a Multi-Layer Perceptron Model :
        Flatten the Training and Testing Data.
        After Flattening Normalize the Data between 0-1
    The problem is a Multi-Class Classification :
        One Hot encode your Labels for both Train and Test
    '''
    # Transform the Input Data
    trainData = dataset.trainData.reshape(dataset.trainData.shape[0], -1).T
    trainData = trainData / 255
    testData = dataset.testData.reshape(dataset.testData.shape[0], -1).T
    testData = testData / 255

    trainLabels = dataset.trainLabel
    # 10 is for number of classes
    trainLabels = one_hot_matrix(trainLabels, 10).T
    testLabels = dataset.testLabel
    # 10 is for number of classes
    testLabels = one_hot_matrix(testLabels, 10).T
    print("Training Data after Flatten and One-Hot Encoding")
    print(trainData.shape, trainLabels.shape)
    print("Testing Data after Flatten and One-Hot Encoding")
    print(testData.shape, testLabels.shape)

    model(trainData=trainData,trainLabels=trainLabels,testData=testData,testLabels=testLabels,
          bacthSize=cfg['batchSize'])