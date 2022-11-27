from loadData import loadDataMultiprocessing
from EdgeDetection import createSketch, createSketchMultiprocessing
import numpy as np
import time
import matplotlib.pyplot as plt
from runModel import runModel
if __name__ == "__main__":
    startLoad = time.time()
    imageTensor = loadDataMultiprocessing()
    endLoad = time.time()
    loadTime = endLoad - startLoad
    #multiprocessing sketches is slower. 
    sketches = createSketch(imageTensor)
    
    model = runModel(imageTensor, sketches)
    
    
    

