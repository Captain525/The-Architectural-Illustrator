import tensorflow as tf
class ConvBlock(tf.keras.Layer):
    def __init__(self, numFilters, BN, Dropout):
        super().__init__()
        self.kernel_size = (4,4)
        self.BN = BN
        self.Dropout = Dropout
        self.stride = (2,2)
        self.numFilters = numFilters

        #WHICH PADDING TO USE?????
        self.padding = "same"
        self.conv = tf.keras.layers.Conv2D(numFilters, self.kernel_size, self.stride, padding = self.padding)
        self.batchNorm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(.5)
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
        """
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride
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
    def __init__(self, numFilters, BN, Dropout):
        super().__init__(self, numFilters, BN, Dropout)
        #only one thing renamed. 
        #PADDING SAME OR NOT????
        #don't know which shape specifically we want. 
        self.outputPadding=  (0,0)

        self.conv = tf.keras.layers.Conv2DTranspose(numFilters, self.kernel_size, self.stride, padding = self.padding, output_padding =self.outputPadding, kernel_initializer = "glorot_normal")

    def calcShape(self, height, width):
        """
        Calculates shape of layer output in this case, it's different than the ConvBlock class. 
        """
        #in case of same padding. In reality more complicated than this, but can't figure it out rn. 
        shape = (self.stride[0]*height, self.stride[1]*width)
        return shape