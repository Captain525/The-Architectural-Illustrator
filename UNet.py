import tensorflow as tf

class UNET(tf.keras.Model):
    """
    UNET is a potential choice for the Generator Architecture as posed by the paper. 
    It's an encoder-decoder architecture with skip connections between layer i in the encoder and layer n-i in the decoder. 
    They concatenate activations from layer i to layer n-i. 
    Ck = Convolution BatchNorm LeakyRelu layer with k filters. With the D, it means dropout of 50%. 
    All convolutions 4x4 spatial filters with stride 2. 
    Convolutions in encoder - downsample by factor of 2. In encoder they updsample by a factor of 2. 
    The encoder structure is: 
    C64-C128-C256-C512-C512-C512-C512-C512
    The decoder structure is: 
    CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    """
    def __init__(self):
        super().__init__()
        self.encblock1 = ConvBlock(64, False, False)
        self.encblock2 = ConvBlock(128, True, False)
        self.encblock3 = ConvBlock(256, True, False)
        self.encblock4 = ConvBlock(512, True, False)
        self.encblock5 = ConvBlock(512, True, False)
        self.encblock6 = ConvBlock(512, True, False)
        self.encblock7 = ConvBlock(512, True, False)
        self.encblock8 = ConvBlock(512, True, False)

        self.decblock1 = ConvTBlock(512, True, True)
        self.decblock2 = ConvTBlock(1024, True, True)
        self.decblock3 = ConvTBlock(1024, True, True)
        self.decblock4 = ConvTBlock(1024, True, False)
        self.decblock5 = ConvTBlock(1024, True, False)
        self.decblock6 = ConvTBlock(512, True, False)
        self.decblock7 = ConvTBlock(256, True, False)
        self.decblock8 = ConvTBlock(128, True, False)
    def call(self, input):
        """
        Include residual connections with the encoder blocks in the decoder. 
        Want connections between layer i and layer n-i. So, layer 7 and 9, 6 and 10 etc. Concatenate along the channels axis. 
        """
        block1 = self.encblock1(input)
        block2 = self.encblock2(block1)
        block3 = self.encblock3(block2)
        block4 = self.encblock4(block3)
        block5 = self.encblock5(block4)
        block6 = self.encblock6(block5)
        block7 = self.encblock7(block6)
        block8 = self.encblock8(block7)

        #finished encoder. 
        block9 = self.decblock1(block8)
        #I think we want to do 7 and 16-7 = 9
        combinedBlock9 = tf.concat(block7, block9, axis=-1)
        block10 = self.decblock2(combinedBlock9)
        combinedBlock10 = tf.concat(block6, block10, axis=-1)
        block11 = self.decblock3(combinedBlock10)
        combinedBlock11 = tf.concat(block5, block11, axis=-1)
        block12 = self.decblock4(combinedBlock11)
        combinedBlock12 = tf.concat(block4, block12, axis=-1)
        block13 = self.decblock5(combinedBlock12)
        combinedBlock13 = tf.concat(block3, block13)
        block14 = self.decblock6(combinedBlock13)
        combinedBlock14 = tf.concat(block2, block14)
        block15 = self.decblock7(combinedBlock14)
        combinedBlock15 = tf.concat(block1, block15)
        block16 = self.decblock8(combinedBlock15)

        #block16 is the output of the UNET, but we add a thing at the end of it as well. 

        return
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
        if self.padding: 

        #hp is the amount of padding. 
            #h0 = (height - 1)*self.stride[0] - hp + self.kernel_size[0]
