import tensorflow as tf
from GAN import GAN
import numpy as np
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import time
from CustomCallbacks import displayImages
def showImages(images):
    for image in images:
        plt.imshow(image)
        plt.show()

def split(images, sketches):
    """
    Splits the data. 
    """
    percentTrain = .8
    numImages = images.shape[0]
    numTrain = int(percentTrain*numImages)
    indices = np.arange(numImages, dtype = int)
    np.random.shuffle(indices)
    print("indices shape: ", indices.shape)
    images = images[indices]
    sketches = sketches[indices]
 
    trainImages = images[:numTrain]
    trainSketches = sketches[:numTrain]
   
    testImages = images[numTrain:]
    testSketches = sketches[numTrain:]
    
    return trainImages, trainSketches, testImages, testSketches



def runModel(images, sketches):
    """
    This method runs the model given the input images and sketches. 
    """
    #maybe use a faster library method for this instead. 
    trainImages, trainSketches, testImages, testSketches = split(images, sketches)
    learningRate = .0002
    b1 = .5
    b2 = .999
    #is giving half the learning rate the same as dividing the objective by 2? 
    optimizerDis = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = b1, beta_2 = b2)
    optimizerGen = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = b1, beta_2 = b2)
    
    batchSize = 4
    epochs = 5
    #our loss function is binary crossentropy. 
    lossFxn = tf.keras.losses.BinaryCrossentropy()

    model = GAN()
    startCompAndBuild = time.time()
    #used to speed up execution potentially. 
    stepsPerExecution = 1
    model.compile(optimizerGen, optimizerDis, lossFxn, lossFxn, metrics = model.createMetrics(), steps_per_execution = stepsPerExecution)
    #need this for eager execution, without this it is automatically not eager. 
    #model.run_eagerly = True
    model.build(input_shape = [(None, 256, 256, 3), (None, 256, 256, 1)])
    endCompAndBuild = time.time()
    compAndBuild = endCompAndBuild - startCompAndBuild
    print("comp and build time: ", compAndBuild)
    model.summary()
    smallerTrainImages = tf.constant(trainImages[:1000], dtype = tf.float32)
    smallerTrainSketches = tf.constant(trainSketches[:1000], dtype = tf.float32)
    smallerTestImages = tf.constant(testImages[:500], dtype = tf.float32)
    smallerTestSketches = tf.constant(testSketches[:500], dtype = tf.float32)
    saveFreq = 10
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint("checkpoints/{epoch}weights", monitor = "sumLoss",save_best_only = True,  mode = "min", save_weights_only = True, save_freq = saveFreq)
    callbacks = [displayImages(smallerTrainImages, smallerTrainSketches, smallerTestImages, smallerTestSketches)]

    history = model.fit(smallerTrainImages, smallerTrainSketches, batch_size = batchSize, epochs = epochs, validation_data = (smallerTestImages, smallerTestSketches), callbacks = callbacks)

    generatedImages = model.generateImages(testSketches[0:10])
    showImages(generatedImages)