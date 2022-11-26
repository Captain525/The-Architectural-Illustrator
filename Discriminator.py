import tensorflow as tf
from PatchGAN import PatchGAN
class Discriminator(tf.keras.Model):
    def __init__(self):
        self.patchGAN = PatchGAN(70)

    def call(self, inputs):
        """
        Integrated conditionality via concatenation. 
        """
        data = inputs[0]
        condition = inputs[1]
        #paper never specified, but figured this out from a source. 
        concatenated = tf.concat(data, condition, axis=-1)
        return self.patchGAN(concatenated)

