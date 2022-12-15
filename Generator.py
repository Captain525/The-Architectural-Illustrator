import tensorflow as tf
from UNet import UNet
from EncDec import EncDec

class Generator(tf.keras.Model):
    """
    Generator for the GAN. 
    We want to find a way to increase the stochasticity of outputs. The paper used dropout to simulate randomness, but 
    it isn't very random. However, they found that the network learns to IGNORE the input noise. we need a way to make the input noise more 
    prominent. 
    """
    def __init__(self, reg_coeff):
        super().__init__()
        self.U = True
        self.l1 = True
        if self.U:
            self.UNet = UNet()
        else:
            self.Unet = EncDec()

        #Regularization coefficient for L1 loss. 
        self.reg_coeff = reg_coeff
        if(self.U):
            self.lastConvolution = tf.keras.layers.Conv2D(3, (4,4), (1,1), padding = "same", activation = "tanh")
        else:
             self.lastConvolution = tf.keras.layers.Conv2DTranspose(3, (4,4), (2,2), padding = "same", activation = "tanh")
        
    def call(self, data, training):
        
        batchSize, height, width, numChannels = data.shape
        uNetOutput = self.UNet(data)
        if self.u:
          assert(uNetOutput.shape == (batchSize, height, width, 128))
        else:
          assert(uNetOutput.shape == (batchSize, int(height/2), int(width/2), 64))
        
        generated = self.lastConvolution(uNetOutput)
        
        assert(generated.shape == (batchSize, height, width, 3))
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
        if(self.l1):
            return lossDefault + self.reg_coeff*l1
        else:
            return lossDefault


