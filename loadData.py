from PIL import Image
from multiprocessing import Process, Pool, Array
import numpy as np
import os
import cv2
import itertools
imageSize = (256,256)
def init(imageArray):
    globals()['imageArray'] = np.frombuffer(imageArray, dtype = int).reshape((numFiles, imageSize[0], imageSize[1], 3))
def loadData():
    dir = "ArchitectureDataset/arcDataset"
    numProcesses = 4
    
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    listDirectories = list(itertools.chain.from_iterable(listDirectories))

    global numFiles
    numFiles = len(listDirectories)
    imageTensor = None
    if __name__ == "__main__":
        print("inside")
        #print("imageSize[0]: ", imageSize[0])
        imageArray = Array('i', np.zeros(numFiles*imageSize[0]*imageSize[0]*3), lock = False)
        
        print("after image array")
        iter = [(i, listDirectories[i]) for i in range(numFiles)]
        #print(iter)
        print('after iter")')
        pool =  Pool(processes = numProcesses, initializer = init, initargs = (imageArray,))
        print(imageArray)
        print("in pool")
        pool.starmap(loadFile,  iter)
        print('after map")')
        
        imageArray = np.frombuffer(imageArray, dtype = int).reshape((numFiles, imageSize[0], imageSize[1], 3))
        
        imageTensor = np.stack(listArray, axis=0)
    if imageTensor is None:
        print("was none")
    return imageTensor
   


def loadFile(fileNum, file_path):
    global imageArray
   #print(fileNum)
    #print(file_path)
    #fileNum = fileNumPath[0]
    #file_path = fileNumPath[1]
    #print("in function")
    im = cv2.imread(file_path)
    imResized = cv2.resize(im, imageSize)
    #print(imResized)
    #print("im resized shape: ", imResized.shape)
    #print("file num: ", fileNum)
    assert(fileNum< numFiles)
    print("image array here: ", imageArray)
    
    imageArray[fileNum] = imResized
    print("done")
   
    return 

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

print("image tensor: ", loadData())