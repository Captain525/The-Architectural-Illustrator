import tensorflow as tf
from GAN import GAN
import numpy as np
from skimage.util import img_as_float
import matplotlib.pyplot as plt
def showImages(images):
    for image in images:
        plt.imshow(image)
        plt.show()
def split(images, sketches):
    percentTrain = .8
    numImages = images.shape[0]
    numTrain = int(percentTrain*numImages)
    indices = np.random.permutation(numImages)
    mixedImages = img_as_float(images[indices])
    mixedSketches = img_as_float(sketches[indices])
    trainImages = mixedImages[:numTrain]
    trainSketches = mixedSketches[:numTrain]

    testImages = mixedImages[numTrain:]
    testSketches = mixedSketches[numTrain:]
    return trainImages, trainSketches, testImages, testSketches



def runModel(images, sketches):
    #maybe use a faster library method for this instead. 
    trainImages, trainSketches, testImages, testSketches = split(images, sketches)
    #showImages(trainSketches[0:10])
    learningRate = .0002
    b1 = .5
    b2 = .999
    optimizerDis = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = b1, beta_2 = b2)
    optimizerGen = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = b1, beta_2 = b2)
    
    batchSize = 6
    epochs = 1
    lossFxn = tf.keras.losses.BinaryCrossentropy()

    model = GAN()

    model.compile(optimizerGen, optimizerDis, lossFxn, lossFxn)
    model.summary()
    model.fit(trainImages, trainSketches, batch_size = batchSize, epochs = epochs, validation_data = (testImages, testSketches))
