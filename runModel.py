import tensorflow as tf
from GAN import GAN


def runModel():
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

    model.fit()
