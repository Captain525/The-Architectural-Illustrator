import numpy as np
import cv2
from skimage.util import img_as_float
from multiprocessing import Array, RawArray, Pool
import ctypes
def createSketch(images):
    """
    numpy array of images, size numImages, imageheight, imageWidth, numChannels. 
    Want to get the outline of each of these images, return them in the same order. 
    This is where we want to replicate a drawing via a gradient
    """
    numImages, height, width, channels = images.shape
    #xGradient gets vertical edges, moves along
    #want the gradient along only the heights and widths of the images. 
    listEdges = []
    for i in range(numImages):
        img = images[i]
        
        edges = cv2.Canny(img, threshold1 = 100, threshold2 = 200, L2gradient = True, apertureSize = 3)[..., np.newaxis]
        #EDGES IS 2D, REMOVES CHANNEL DIMENSION
        #does it have the same number of channels? 
        assert(edges.shape[0:2] == img.shape[0:2])
        listEdges.append(edges)
    
    #UINT 8 CURRENTLY. Convert from float range 0 to 1
    gradientImages = img_as_float(np.array(listEdges))
    assert((gradientImages<=1.0).all())
    assert((gradientImages>=0.0).all())
    print("dtype: ", gradientImages.dtype)
    #might be a different number of channels. 
    assert(gradientImages.shape[0:3] == images.shape[0:3])
    return gradientImages

def createSketchMultiprocessing(images):
    numImages, height, width, channels = images.shape
    numProcesses = 4

    sketchArray = RawArray(ctypes.c_char, numImages*height*width*channels)
    iter = [(i, images[i]) for i in range(numImages)]

    with Pool(processes = numProcesses, initializer = init, initargs = (sketchArray, numImages, height, width, channels)) as pool:
        pool.starmap(sketch, iter)
    sketchArray = np.frombuffer(sketchArray, dtype = np.uint8).reshape((numImages, height, width, channels))
    return sketchArray
def init(sketchArray, numImages, height, width, channels):
    buffered = np.frombuffer(sketchArray, dtype = np.uint8)
    globals()['sketchArray'] = buffered.reshape((numImages, height, width, channels))
    return

def sketch(num, image):
    global sketchArray
    edges = cv2.Canny(image, threshold1 = 100, threshold2 = 200, L2gradient = True, apertureSize = 3)[..., np.newaxis]
    sketchArray[num] = edges
    return