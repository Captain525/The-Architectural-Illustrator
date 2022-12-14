import tensorflow as tf
from PatchGAN import PatchGAN, NotAPatchGAN
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.patch = True
        if self.patch:
            self.patchGAN = PatchGAN(16)
        else:
            self.patchGAN = NotAPatchGAN(16)

    def call(self, inputs, training):
        """
        Integrated conditionality via concatenation. 
        """
        data = inputs[0]
        condition = inputs[1]
        #Concatenate the data to the condition to take both into account in the network. 
        concatenated = tf.concat([data, condition], axis=-1)
        return self.patchGAN(concatenated)

    def compute_loss(self, combined, predGen, predReal, sample_weights=None):
        """
        Generates the labels and computes the loss. 
        Computes the two losses separately and then sums them together. 
        """
        #these aren't really used here. 
        generated = combined[0]
        x = combined[1]
        if(self.patch):
          assert(predGen.shape == (generated.shape[0], 32, 32, 1))
        else:
            assert(predGen.shape == (generated.shape[0], 1))
       
        #THIS CODE WORKS FOR 1D OUTPUT AND PATCH GAN OUTPUT. 
        #generate the labels here instead of earlier before. 
        realLabels = tf.cast(tf.logical_not(tf.cast(0*predReal, bool)), tf.int32)
        genLabels = tf.cast(0*predGen, tf.int32)

        realLoss = self.compiled_loss(realLabels, predReal, sample_weights)
        genLoss = self.compiled_loss(genLabels, predGen, sample_weights)
        return realLoss+genLoss