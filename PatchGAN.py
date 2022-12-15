import tensorflow as tf
from Blocks import ConvBlock, ConvTBlock
class PatchGAN(tf.keras.layers.Layer):
    """
    Layer called from the discriminator model. Pretty much does all the work of the discriminator though. Could make this a model. 

    Basically performs classification on each patch of the input image as real or fake. Then, you calculate the loss over that grid
    instead of just one classification value. 
    """
    def __init__(self, size, normalConvolution):
        super().__init__()
        self.normalConvolution = normalConvolution
        #assuming we have a (size,size) discriminator

        #FORMULA FOR RECEPTIVE FIELD SIZE: 
        #r = sum l= 1 to L (kl-1)*Prod i = 1 to l-1 (si) + 1
        #for our example, since kl = 4, si = 2 for all layers except the last, and
        
        #specifically for a model structure like this, with n -1 conv blocks of (4,4) and 2 stride, 
        #and one final layer of stride 1 kernel 4, the formula is: 
        # r = -2 + 9*2^(L-2)
        
        self.conv1 = ConvBlock(64, False, False)
        self.conv2 = ConvBlock(128, True, False)

        """
        #add this if want 70x70 patch gan. 
        #self.conv3 = ConvBlock(256, True, False)
        #self.conv4 = ConvBlock(512, True, False)
        """
        kernelInitializer = tf.keras.initializers.RandomNormal(mean= 0, stddev = .02)
        #forgot a padding = same on first run
        self.lastConvLayer = tf.keras.layers.Conv2D(1, kernel_size = (4,4), strides = (2,2), padding = "same", activation = "sigmoid", kernel_initializer=kernelInitializer)
        if(self.normalConvolution):
            self.lastConvLayer = tf.keras.layers.Conv2D(1, kernel_size = (4,4), strides = (2,2), padding = "same", kernel_initializer = kernelInitializer)
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(1, activation = "sigmoid")
    def call(self, input):
        #print("PG input shape: ", input.shape)
        layer1 = self.conv1(input)
        #print("PG layer1 shape: ", layer1.shape)
        layer2 = self.conv2(layer1)
        """
        #add this if want 70x70 patch gan. 
        #print("PG layer2 shape: ", layer2.shape)
        #layer3 = self.conv3(layer2)
        #print("PG layer3 shape: ", layer3.shape)
        #layer4 = self.conv4(layer3)
        ##print("PG layer4 shape: ", layer4.shape)
        """

        #this has the shape of batchSize x numPatches x numPatches x 1
        output = self.lastConvLayer(layer2)
        if self.normalConvolution: 
            flattened = self.flatten(output)
            assert(flattened.shape == (output.shape[0], output.shape[1]*output.shape[2]))
            output = self.dense(flattened)
            assert(output.shape == (output.shape[0], 1))
        
        return output
        
class NotAPatchGAN(tf.keras.Layer):
    def __init__(self, size):
        super().__init__()
        #assuming we have a (size,size) discriminator

        #FORMULA FOR RECEPTIVE FIELD SIZE: 
        #r = sum l= 1 to L (kl-1)*Prod i = 1 to l-1 (si) + 1
        #for our example, since kl = 4, si = 2 for all layers except the last, and
        
        #specifically for a model structure like this, with n -1 conv blocks of (4,4) and 2 stride, 
        #and one final layer of stride 1 kernel 4, the formula is: 
        # r = -2 + 9*2^(L-2)
        kernelInitializer = tf.keras.initializers.RandomNormal(mean= 0, stddev = .02)
        self.conv1 = ConvBlock(64, False, False)
        self.conv2 = ConvBlock(128, True, False)
        self.lastConvLayer = tf.keras.layers.Conv2D(1, kernel_size = (4,4), strides = (2,2), padding = "same",  kernel_initializer=kernelInitializer)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation = "sigmoid")

      

    def call(self, input):
     
        output = self.conv1(input)
     
        output = self.conv2(output)

        output = self.lastConvLayer(output)
        #this has the shape of batchSize x numPatches x numPatches x 1 
        output = self.flatten(output)
        output = self.dense(output)

        
        return output