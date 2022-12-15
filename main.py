from loadData import loadDataMultiprocessing, loadData, splitIntoFolders, renameFiles
from EdgeDetection import *
import numpy as np
import time
import matplotlib.pyplot as plt
from runModel import runModel
"""
RUN THE MODEL FROM HERE POTENTIALLY. This method is essential to use the multiprocessing. 
"""
if __name__ == "__main__":

    startLoad = time.time()
    folderLengths, imageTensor = loadDataMultiprocessing()
    endLoad = time.time()
    loadTime = endLoad - startLoad
    print("Load time: ", loadTime)
    
    
    #multiprocessing sketches is slower. 
    startSketch = time.time()
    sketches = createSketch(imageTensor)
    endSketch = time.time()
    sketchTime = endSketch - startSketch
    print("Sketch time: ", sketchTime)

    model = runModel(imageTensor, sketches)
    
    
    

