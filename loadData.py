from PIL import Image
from multiprocessing import Process, Pool, Array, RawArray
import numpy as np
import os
import cv2
import itertools
import time
import ctypes
imageSize = (256,256)
def init(imageArray, len, str):
    buffered = np.frombuffer(imageArray, dtype = np.uint8)
    size = len*imageSize[0]*imageSize[1]*3
    globals()[str] = buffered.reshape((len, imageSize[0], imageSize[1], 3))
    return 
def loadFile(fileNum, file_path):
    global imageArray
    im = cv2.imread(file_path)
    imResized = cv2.resize(im, imageSize)
    imageArray[fileNum] = imResized
    return 
def loadFileDiff(fileNum, file_path):
    global imageList
    im = cv2.imread(file_path)
    imResized = cv2.resize(im, imageSize)
    imageList[fileNum] = imResized
def loadDataMultiprocessing():
   
    dir = "ArchitectureDataset/arcDataset"
    numProcesses = 4
    #gets a list of ALL files in one. 
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    lenLists = [len(folder) for folder in listDirectories]
    print(len(lenLists))
    listDirectories = list(itertools.chain.from_iterable(listDirectories))
    imageArray = []
    global numFiles
    numFiles = len(listDirectories)
    firstSection = listDirectories
    
    imageArray = RawArray(ctypes.c_char, len(firstSection)*imageSize[0]*imageSize[1]*3)

        
    
    iter = [(i, listDirectories[i]) for i in range(len(firstSection))]
    
    with Pool(processes = numProcesses, initializer = init, initargs = (imageArray, len(firstSection), 'imageArray')) as pool:
        
        pool.starmap(loadFile,  iter)
    imageArray = np.frombuffer(imageArray, dtype = np.uint8).reshape((len(firstSection), imageSize[0], imageSize[1], 3))
    return lenLists, imageArray
def splitIntoFolders(imageTensor, lenFolders):
    print("got into split folders")
    listFolders = []
    currentIndex = 0
    for length in lenFolders:
        subsetFolder = imageTensor[currentIndex:currentIndex + length, :]
        listFolders.append(subsetFolder)
    return listFolders
"""
def loadDatabyCategory():
    dir = "ArchitectureDataset/arcDataset"
    
    listDirectories = [[x[0] + "/"+  file for file in x[2]] for x in os.walk(dir)][1:]
    listFolders = []
    for list in listDirectories:

        listFolders.append(multiprocessingCalcList(list))
        print("done one")
    return listFolders
"""
def multiprocessingCalcList(list):
    """
    Multiprocessing per folder in the file. 
    """
    global numListEles
    numProcesses = 4
    numListEles = len(list)

    imageList = RawArray(ctypes.c_char, numListEles*imageSize[0]*imageSize[1]*3)

    iter = [(i, list[i]) for i in range(numListEles)]
    with Pool(processes = numProcesses, initializer = init, initargs=(imageList, numListEles, 'imageList')) as pool:
        pool.starmap(loadFileDiff, iter)
    imageList = np.frombuffer(imageList, dtype = np.uint8).reshape((numListEles, imageSize[0], imageSize[1], 3))
    return imageList

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

   
    
    