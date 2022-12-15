import numpy as np
import cv2
from skimage.util import img_as_float
import tensorflow as tf
from multiprocessing import Array, RawArray, Pool
from runModel import showImages
import ctypes
import matplotlib.pyplot as plt
def showImageGroups(listImageArrays, size):
    """
    Shows one image from each list. 
    """
    numImages= len(listImageArrays)
    fig = plt.figure(figsize=(10, 10))
    rows = size
    columns = numImages

    for i in range(0, rows):
            for j in range(columns):
                index = i*columns + j + 1
                fig.add_subplot(rows, columns, index)
                plt.imshow(listImageArrays[j][i, :, :], aspect = 'auto')  
                plt.axis('off')
    plt.subplots_adjust(hspace=0, wspace = 0)
    plt.show()
    return
def outlineAlgorithm(sketches):
    """
    Goal: given an edge detected image, find the outline of that sketch. 
    THIS METHOD ISN'T USED IN THE FINAL RESULT. Experimenting with finding the outline. 
    
    """
    kernelSize = 10
    print("in outline algorithm")
    print(sketches.shape)
    kernel = np.ones(shape = (kernelSize, kernelSize, sketches.shape[-1]))/(kernelSize*kernelSize*sketches.shape[-1])
    listSketches =[]
    listValues = []
    for sketch in sketches:
        sketch = cv2.filter2D(sketch, -1, kernel = kernel)[..., np.newaxis]
        value = cv2.Canny((255*np.logical_and(sketch<.3*256,sketch>0.1*256)).astype(np.uint8), threshold1 = 1, threshold2 = 10, apertureSize = 3)
        showImages(value[np.newaxis,...])
        showImages(sketch[np.newaxis,...])
        listSketches.append(sketch)
        listValues.append(value)
    newSketches= np.stack(listSketches, axis=0)
    outlinesNow = createSketch(newSketches)
    listInput = [sketches, newSketches, outlinesNow]
    showImageGroups(listInput, 5)
    return sketches

def randomJitter(images):
    """
    NOT USED IN FINAL RESULT. 
    Randomly jitters images, by increasing their size then cropping them. 

    Not sure if for loop is fastest way to do this. 
    """
    print("images shape: ", images.shape)
    resizedImages = tf.image.resize(images, size = (286,286))
    print("resized shape: ", resizedImages.shape)
    listRandomCrop = []
    for i in range(images.shape[0]):
        randomCropping = tf.image.random_crop(resizedImages[i], (256,256, resizedImages.shape[-1]))
        listRandomCrop.append(randomCropping)
    croppedImages = tf.stack(listRandomCrop, axis=0)
    return croppedImages.numpy().astype(np.uint8)

def createSketch(images):
    """
    numpy array of images, size numImages, imageheight, imageWidth, numChannels. 
    Want to get the outline of each of these images, return them in the same order. 
    This is where we want to replicate a drawing via a gradient. 
    """
    numImages, height, width, channels = images.shape
    #xGradient gets vertical edges, moves along
    #want the gradient along only the heights and widths of the images. 
    listEdges = []
    for i in range(numImages):
        img = images[i]
        #100, 200 works very well but gets TOO much detail. 
        minVal = 100
        maxVal = 200
        edges = cv2.Canny(img, threshold1 = minVal, threshold2 = maxVal, L2gradient = True, apertureSize = 3)[..., np.newaxis]
        #EDGES IS 2D, REMOVES CHANNEL DIMENSION
        #does it have the same number of channels? 
        assert(edges.shape[0:2] == img.shape[0:2])
        listEdges.append(edges)
    
    #UINT 8 CURRENTLY. Convert from float range 0 to 1
    gradientImages = img_as_float(np.stack(listEdges, axis=0))
    gradientImages = np.stack(listEdges, axis=0)
    assert((gradientImages<=1.0).all())
    assert((gradientImages>=0.0).all())

    #might be a different number of channels. 
    assert(gradientImages.shape[0:3] == images.shape[0:3])
    return gradientImages

def createSketchMultiprocessing(images):
    """
    this was actually slower than the normal create sketch method so we don't use this. 
    """
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