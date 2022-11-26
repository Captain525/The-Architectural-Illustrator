from PIL import Image
from multiprocessing import Process, Pool, Array
import numpy as np
import os
import cv2
import itertools
imageSize = (256,256)
def loadData():
    dir = "ArchitectureDataset/arcDataset"
    numProcesses = 4
    
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    listDirectories = list(itertools.chain.from_iterable(listDirectories))

    
    numFiles = len(listDirectories)
    
    if __name__ == '__main__':
        imageArray = Array('u', np.zeros(shape = (numFiles, imageSize[0], imageSize[1], 3)))
        iter = ((i, listDirectories[i], imageArray) for i in range(numFiles))
        with Pool() as pool: 
            pool.map_async(loadFile, iterable = iter)
            pool.close()
            pool.join()
        imageTensor = np.array(imageArray, dtype = np.uint8)
        return imageTensor
    return

def loadFile(fileNum, file_path, imageArray):
    im = cv2.imread(file_path)
    imResized = cv2.resize(im, imageSize)
    imageArray[fileNum] = imResized

def loadFiles(n, numProcesses, numIterations, listDirectories):
    listImageTensors = []
    for i in range(0, numIterations):
        index = n + numProcesses*i
        directory, listFiles = listDirectories[index]
        imageTensor = loadFilesFromDirectory(directory, listFiles)
        listImageTensors.append(imageTensor)
    imageTensorCombined = np.concatenate(listImageTensors, axis=0)
    return imageTensorCombined

def loadFilesFromDirectory(dir, listFiles):
    iterator = (dir + "/" + path for path in listFiles)
    listImages = []
    numImages= len(listFiles)
    for path in iterator:
        #which type is it? 
        im = cv2.imread(path)
        imResized = cv2.resize(im, imageSize)
        listImages.append(imResized)
    imageTensor = np.stack(listImages, axis=0)
    assert(imageTensor.shape == (numImages, imageSize[0], imageSize[1], 3))
    return imageTensor


print(loadData())