from loadData import loadDataMultiprocessing, loadData, splitIntoFolders
from EdgeDetection import createSketch, createSketchMultiprocessing, randomJitter
import numpy as np
import time
import matplotlib.pyplot as plt
from runModel import runModel
if __name__ == "__main__":
    startLoad = time.time()
    folderLengths, imageTensor = loadDataMultiprocessing()
    endLoad = time.time()
    loadTime = endLoad - startLoad
    print("Load time: ", loadTime)
    #startSplit = time.time()
    #listFolders = splitIntoFolders(imageTensor, folderLengths)
    #endSplit = time.time()
    #splitTime = endSplit-startSplit
    #print("split time: ", splitTime)
    #imageTensor = randomJitter(imageTensor)
    
    #multiprocessing sketches is slower. 
    startSketch = time.time()
    sketches = createSketch(imageTensor)
    endSketch = time.time()
    sketchTime = endSketch - startSketch
    print("Sketch time: ", sketchTime)

    model = runModel(imageTensor, sketches)
    
    
    

