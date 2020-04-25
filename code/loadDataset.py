import numpy as np
import glob
import os
import cv2

class loadDataset:
    def __init__(self,cfg):
        self.trainData , self.trainLabel = self.readData(cfg['trainDir'])
        self.testData , self.testLabel = self.readData(cfg['testDir'])

    def readData (self,inputDir):
        data , labels = [] , []
        for dir in os.listdir(inputDir):
            imagesPathList = glob.glob(os.path.join(inputDir,dir,"*.png"))
            for imagePath in imagesPathList:
                cvImage = cv2.imread(imagePath)
                data.append(cvImage)
                labels.append(int(dir))
        data = np.array(data)
        labels = np.array(labels)
        return data , labels
