import tensorflow as tf
from GAN import GAN
import numpy as np
from skimage.util import img_as_float
def split(images, sketches):
    percentTrain = .8
    numImages = images.shape[0]
    numTrain = int(percentTrain*numImages)
    indices = np.random.permutation(numImages)
    mixedImages = img_as_float(images[indices])
    mixedSketches = img_as_float(sketches[indices])
    trainImages = images[:numTrain]
    trainSketches = sketches[:numTrain]

    testImages = images[numTrain:]
    testSketches = sketches[numTrain:]
    return trainImages, trainSketches, testImages, testSketches



def runModel(images, sketches):
    #maybe use a faster library method for this instead. 
    trainImages, trainSketches, testImages, testSketches = split(images, sketches)
    
    learningRate = .0002
    b1 = .5
    b2 = .999
    optimizerDis = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = b1, beta_2 = b2)
    optimizerGen = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = b1, beta_2 = b2)
    
    batchSize = 6
    epochs = 1
    lossFxn = tf.keras.losses.BinaryCrosssentropy()

    model = GAN()

    model.compile(optimizerGen, optimizerDis, lossFxn, lossFxn)

    model.fit(trainImages, trainSketches, batch_size = batchSize, epochs = epochs, validation_data = (testImages, testSketches))
    