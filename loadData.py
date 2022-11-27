from PIL import Image
from multiprocessing import Process, Pool, Array, RawArray
import numpy as np
import os
import cv2
import itertools
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
    
    print("got imageArray", flush =True)
    
    imageArray[fileNum] = imResized
    print("done", flush=True)
   
    return 
def loadDataMultiprocessing():

    dir = "ArchitectureDataset/arcDataset"
    numProcesses = 4
    
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    listDirectories = list(itertools.chain.from_iterable(listDirectories))

    global numFiles
    numFiles = len(listDirectories)
    firstSection = listDirectories[0:10000]
    imageTensor = None
    if __name__ == '__main__':
        print("__name__: ", __name__)
        print("inside")
        #print("imageSize[0]: ", imageSize[0])
        print(numFiles*imageSize[0]*imageSize[1]*3)
        imageArray = RawArray(ctypes.c_char_p,len(firstSection*imageSize[0]*imageSize[1]*3))
        
        print("after image array")
        iter = [(i, listDirectories[i]) for i in range(firstSection.shape)]
        #print(iter)
       # with Pool(processes = numProcesses, initializer = init, initargs = (imageArray,)) as pool:
            #print("inside pool", flush = True)
            #pool.map(loadFile,  iter)
        with Pool() as pool:
            print(pool.map(loadFile, iter))
          

        imageArray = np.frombuffer(imageArray, dtype = int).reshape((numFiles, imageSize[0], imageSize[1], 3))
        print(imageArray)
    print(globals()['imageArray'])
    exit()
    return globals()['imageArray']
        
 
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

if __name__ == '__main__':
    dir = "ArchitectureDataset/arcDataset"
    numProcesses = 4
    
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    listDirectories = list(itertools.chain.from_iterable(listDirectories))

    global numFiles
    numFiles = len(listDirectories)
    firstSection = listDirectories[0:1000]
    print("__name__: ", __name__)
    print("inside")
    #print("imageSize[0]: ", imageSize[0])
    print(numFiles*imageSize[0]*imageSize[1]*3)
    size = len(firstSection)*imageSize[0]*imageSize[1]*3
    imageArray = RawArray(ctypes.c_char, len(firstSection)*imageSize[0]*imageSize[1]*3)

        
    print("after image array")
    iter = [(i, listDirectories[i]) for i in range(len(firstSection))]
    #print(iter)
    with Pool(processes = numProcesses, initializer = init, initargs = (imageArray, len(firstSection))) as pool:
        print("inside pool", flush = True)
        pool.starmap(loadFile,  iter)
    #with Pool() as pool:
        #print(pool.map(loadFile, iter))
          
    print("here")
    imageArray = np.frombuffer(imageArray, dtype = np.uint8).reshape((len(firstSection), imageSize[0], imageSize[1], 3))
    print(imageArray)
    exit()