from PIL import Image
from multiprocessing import Process, Pool, Array, RawArray
import numpy as np
import os
import cv2
import itertools
import time
import ctypes
imageSize = (256,256)
def init(imageArray, len):
   
    buffered = np.frombuffer(imageArray, dtype = np.uint8)
    #print(buffered.shape, flush = True)
    #print("len: ", len)
    size = len*imageSize[0]*imageSize[1]*3
    #print(buffered.shape[0]/size)
    globals()['imageArray'] = buffered.reshape((len, imageSize[0], imageSize[1], 3))
    return 
   
        

    
def loadFile(fileNum, file_path):
    global imageArray
    im = cv2.imread(file_path)
    imResized = cv2.resize(im, imageSize)
    imageArray[fileNum] = imResized
    return 
def loadDataMultiprocessing():
    
    dir = "ArchitectureDataset/arcDataset"
    numProcesses = 4
    
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    listDirectories = list(itertools.chain.from_iterable(listDirectories))
    imageArray = []
    global numFiles
    numFiles = len(listDirectories)
    firstSection = listDirectories
    
    
    #print("imageSize[0]: ", imageSize[0])
    print(numFiles*imageSize[0]*imageSize[1]*3)
    size = len(firstSection)*imageSize[0]*imageSize[1]*3
    imageArray = RawArray(ctypes.c_char, len(firstSection)*imageSize[0]*imageSize[1]*3)

        
    
    iter = [(i, listDirectories[i]) for i in range(len(firstSection))]
    
    with Pool(processes = numProcesses, initializer = init, initargs = (imageArray, len(firstSection))) as pool:
        
        pool.starmap(loadFile,  iter)
    imageArray = np.frombuffer(imageArray, dtype = np.uint8).reshape((len(firstSection), imageSize[0], imageSize[1], 3))
    return imageArray
        
 
def loadData():
    dir = "ArchitectureDataset/arcDataset"
    numProcesses = 4
    
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    listDirectories = list(itertools.chain.from_iterable(listDirectories))

    global numFiles
    numFiles = len(listDirectories)
    listImages = []
    for i in range(numFiles):
        listImages.append(loadFileNormal(listDirectories[i]))
    imageTensor = np.stack(listImages, axis = 0)
    assert(imageTensor.shape == (numFiles, imageSize[0], imageSize[1], 3))
    return imageTensor
def loadFileNormal(directory):
    im = cv2.imread(directory)
    imResized = cv2.resize(im ,imageSize)
    return imResized

if __name__ == "__main__":
    
    imageTensor = loadDataMultiprocessing()
   
    
    