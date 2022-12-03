import tensorflow as tf
from GAN import GAN
import numpy as np
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import time
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
    #is giving half the learning rate the same as dividing the objective by 2? 
    optimizerDis = tf.keras.optimizers.Adam(learning_rate = learningRate/2, beta_1 = b1, beta_2 = b2)
    optimizerGen = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = b1, beta_2 = b2)
    
    batchSize = 4
    epochs = 5
    lossFxn = tf.keras.losses.BinaryCrossentropy()

    model = GAN()
    startCompAndBuild = time.time()
    model.compile(optimizerGen, optimizerDis, lossFxn, lossFxn)
    #need this for eager execution, without this it is automatically not eager. 
    #model.run_eagerly = True
    model.build(input_shape = [(None, 256, 256, 3), (None, 256, 256, 1)])
    endCompAndBuild = time.time()
    compAndBuild = endCompAndBuild - startCompAndBuild
    print("comp and build time: ", compAndBuild)
    model.summary()
    print("ready to train")
    smallerTrainImages = tf.constant(trainImages[:1000], dtype = tf.float32)
    smallerTrainSketches = tf.constant(trainSketches[:1000], dtype = tf.float32)
    smallerTestImages = tf.constant(testImages[:500], dtype = tf.float32)
    smallerTestSketches = tf.constant(testSketches[:500], dtype = tf.float32)
    saveFreq = 10
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint("/tmp/checkpoint", monitor = "valSumLoss",save_best_only = False,  mode = "min", save_weights_only = False, save_freq = saveFreq)
    callbacks = [modelCheckpoint]
    history = model.fit(smallerTrainImages, smallerTrainSketches, batch_size = batchSize, epochs = epochs, validation_data = (smallerTestImages, smallerTestSketches), callbacks = callbacks)

    generatedImages = model.generateImages(trainSketches[0:10])
    showImages(generatedImages)