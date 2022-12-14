import tensorflow as tf
class ConvBlock(tf.keras.layers.Layer):
    """
    Defines a block used in the UNET and PatchGAN architectures. 
    It consists of a Conv2D layer with kernel size 4 and stride 2, then followed by batchNorm, maybe dropout, then RELU. 
    This is also the super class for the transpose block. 

    Dropout is 50%, Batch normalization is 
    
    """
    def __init__(self, numFilters, BN, Dropout):
        super().__init__()
        self.kernel_size = (4,4)
        self.BN = BN
        self.Dropout = Dropout
        self.stride= (2,2)
        self.numFilters = numFilters

        self.padding = "same"
        kernelInitializer = tf.keras.initializers.RandomNormal(mean=0, stddev = .02)
        self.conv = tf.keras.layers.Conv2D(numFilters, self.kernel_size, self.stride, padding = self.padding, kernel_initializer = kernelInitializer)
        self.batchNorm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(.5)
        #used just leaky ReLU. 
        self.relu = tf.keras.layers.LeakyReLU(.2)

    def call(self, input):
        
        batchSize, height, width, numChannels = input.shape
        convOutput = self.conv(input)
        newHeight, newWidth = self.calcShape(height, width)
       
        assert(convOutput.shape == (batchSize, newHeight, newWidth, self.numFilters))
        if(self.BN):
            convOutput = self.batchNorm(convOutput)
        if(self.Dropout):
            convOutput = self.dropout(convOutput)
        activated = self.relu(convOutput)
        return activated

    def calcShape(self, height, width):
        """
        Calculates the shape of the output of this layer given the input shape. 
        Used to check if the convBlock is working as intended. 
        """
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.stride       ## filter stride
        # Cleaning padding input.
        ry = height%sh
        rx = width %sw
        if(self.padding == "same"):
            valueHeight = fh- ry - sh*int(not ry)
            heightPad = max(valueHeight, 0)
            #same here. 
            valueWidth = fw-rx -sw*int(not rx)
            widthPad = max(valueWidth, 0)
            #heightPad and width pad are total amount you should pad, so get left and right pad here. 
        else:
            heightPad, widthPad = 0,0
        outputHeight = (height + heightPad - fh)//sh + 1
        outputWidth = (width + widthPad - fw)//sw + 1
        return outputHeight, outputWidth
class ConvTBlock(ConvBlock):
    """
    Class for the ConvTBlock in the decoders. This upsamples the image, doubling the size. 
    """
    def __init__(self, numFilters, BN, Dropout):
        super().__init__(numFilters, BN, Dropout)
        #only one thing renamed. 
        #no output padding gets the right results. 
        kernelInitializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = .02)
        self.conv = tf.keras.layers.Conv2DTranspose(numFilters, self.kernel_size, self.stride, padding = self.padding,  kernel_initializer = kernelInitializer)
   
    def calcShape(self, height, width):
        """
        Calculates shape of layer output in this case, it's different than the ConvBlock class. 
        """
        #in case of same padding. In reality more complicated than this, but can't figure it out rn. 
        shape = (self.stride[0]*height, self.stride[1]*width)
        return shape