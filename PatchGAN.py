import tensorflow as tf
from Blocks import ConvBlock, ConvTBlock
class PatchGAN(tf.keras.layers.Layer):
    """
    Should this class be a model or a layer? 
    """
    def __init__(self, size):
        super().__init__()
        #assuming we have a (size,size) discriminator

        #FORMULA FOR RECEPTIVE FIELD SIZE: 
        #r = sum l= 1 to L (kl-1)*Prod i = 1 to l-1 (si) + 1
        #for our example, since kl = 4, si = 2 for all layers except the last, and
        
        #specifically for a model structure like this, with n -1 conv blocks of (4,4) and 2 stride, 
        #and one final layer of stride 1 kernel 4, the formula is: 
        # r = -2 + 9*2^(L-2)
        
        #70 x70 architecture:
        self.conv1 = ConvBlock(64, False, False)
        self.conv2 = ConvBlock(128, True, False)
        #self.conv3 = ConvBlock(256, True, False)
        #self.conv4 = ConvBlock(512, True, False)
        kernelInitializer = tf.keras.initializers.RandomNormal(mean= 0, stddev = .02)
        #forgot a padding = same on first run
        self.lastConvLayer = tf.keras.layers.Conv2D(1, kernel_size = (4,4), strides = (2,2), padding = "same", activation = "sigmoid", kernel_initializer=kernelInitializer)
    def call(self, input):
        batchSize = input.shape[0]
        print("PG input shape: ", input.shape)
        layer1 = self.conv1(input)
        print("PG layer1 shape: ", layer1.shape)
        layer2 = self.conv2(layer1)
        print("PG layer2 shape: ", layer2.shape)
        #layer3 = self.conv3(layer2)
        #print("PG layer3 shape: ", layer3.shape)
        #layer4 = self.conv4(layer3)
        ##print("PG layer4 shape: ", layer4.shape)
        #maybe add this part to the discriminator class instead. 
        output = self.lastConvLayer(layer2)
        print("Output shape:  ", output.shape)
        
        return output
        
