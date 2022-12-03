import tensorflow as tf
from UNet import UNet

class Generator(tf.keras.Model):
    """
    Generator for the GAN. 
    We want to find a way to increase the stochasticity of outputs. The paper used dropout to simulate randomness, but 
    it isn't very random. However, they found that the network learns to IGNORE the input noise. we need a way to make the input noise more 
    prominent. 
    """
    def __init__(self, reg_coeff):
        super().__init__()
        self.UNet = UNet()
        #Regularization coefficient for L1 loss. 
        self.reg_coeff = reg_coeff
        self.lastConvolution = tf.keras.layers.Conv2D(3, (4,4), (1,1), padding = "same", activation = "tanh")
        #range -1 to 1. 
        self.tanh = tf.keras.activations.tanh
    def call(self, data, training):
        
        batchSize, height, width, numChannels = data.shape
        uNetOutput = self.UNet(data)
        assert(uNetOutput.shape == (batchSize, height, width, 128))
        generated = self.lastConvolution(uNetOutput)
        #trying to put in range 0, 1
        #generated = (self.tanh(generated) + 1)/2
        assert(generated.shape[0:3] == data.shape[0:3])
        return generated

    def compute_loss(self, combined,  genPred, genReal, sample_weight=None):
        """
        Want binary crossentropy with L1 regularization. 
        """
        generated = combined[0]
        x = combined[1]
        difference = generated-x
        realY = tf.cast(tf.logical_not(tf.cast(0*genPred, bool)), tf.int32)
        #calls the loss function passed into the compiler. 
        lossDefault = self.compiled_loss(realY, genPred, sample_weight)
        #penalty gets a scalar value instead of a batchSize tensor. 
        l1 = tf.reduce_sum(tf.abs(difference), axis = [1,2,3])
        return lossDefault + self.reg_coeff*l1


