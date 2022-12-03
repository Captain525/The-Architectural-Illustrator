import tensorflow as tf
from PatchGAN import PatchGAN
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.patchGAN = PatchGAN(16)

    def call(self, inputs, training):
        """
        Integrated conditionality via concatenation. 
        """
        data = inputs[0]
        condition = inputs[1]
        #paper never specified, but figured this out from a source. 
        concatenated = tf.concat([data, condition], axis=-1)
        return self.patchGAN(concatenated)

    def compute_loss(self, X, predGen, predReal, sample_weights):
        #all ones. 
        realLabels = tf.cast(tf.logical_not(tf.cast(0*predReal, bool)), tf.int32)
        genLabels = tf.cast(0*predGen, tf.int32)

        realLoss = self.compiled_loss(realLabels, predReal, sample_weights)
        genLoss = self.compiled_loss(genLabels, predGen, sample_weights)
        return realLoss+genLoss