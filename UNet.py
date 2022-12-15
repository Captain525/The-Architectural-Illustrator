import tensorflow as tf
from Blocks import ConvBlock, ConvTBlock
class UNet(tf.keras.layers.Layer):
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

    Oour smaller UNet has the structure. 
    CD512-CD512-CD512 - C512 -C512-C256-C128-C128

    """
    def __init__(self):
        super().__init__()
        #If True, UNET II, if False, UNET I
        self.small = False
        
        self.encblock1 = ConvBlock(64, False, False)
        self.encblock2 = ConvBlock(128, True, False)
        self.encblock3 = ConvBlock(256, True, False)
        self.encblock4 = ConvBlock(512, True, False)
        self.encblock5 = ConvBlock(512, True, False)
        self.encblock6 = ConvBlock(512, True, False)
        self.encblock7 = ConvBlock(512, True, False)
        self.encblock8 = ConvBlock(512, True, False)

        self.decblock1 = ConvTBlock(512, True, True)
        #2 if not small 1 if small. 
        value = int(not self.small) + 1
        #in case of big block ie UNET I, all these values are doubled. 
        self.decblock2 = ConvTBlock(512*value, True, True)
        self.decblock3 = ConvTBlock(512*value, True, True)
        self.decblock4 = ConvTBlock(512*value, True, False)
        self.decblock5 = ConvTBlock(512*value, True, False)
        self.decblock6 = ConvTBlock(256*value, True, False)
        self.decblock7 = ConvTBlock(128*value, True, False)
       
        self.decblock8 = ConvTBlock(128, True, False)
        
    def call(self, input):
        """
        Include residual connections with the encoder blocks in the decoder. 
        Want connections between layer i and layer n-i. So, layer 7 and 9, 6 and 10 etc. Concatenate along the channels axis. 
        """
        #print("UNET input shape: ", input.shape)
        block1 = self.encblock1(input)
        #print("UNET block 1 shape: ", block1.shape)
        block2 = self.encblock2(block1)
        #print("UNET block 2 shape: ", block2.shape)
        block3 = self.encblock3(block2)
        #print("UNET block 3 shape: ", block3.shape)
        block4 = self.encblock4(block3)
        #print("UNET block 4 shape: ", block4.shape)
        block5 = self.encblock5(block4)
        #print("UNET block 5 shape: ", block5.shape)
        block6 = self.encblock6(block5)
        #print("UNET block 6 shape: ", block6.shape)
        block7 = self.encblock7(block6)
        #print("UNET block 7 shape: ", block7.shape)
        block8 = self.encblock8(block7)
        #print("UNET block 8 shape: ", block8.shape)

        #finished encoder. 
        block9 = self.decblock1(block8)
        #print("UNET block 9 shape: ", block9.shape)
        #I think we want to do 7 and 16-7 = 9
        combinedBlock9 = tf.concat([block7, block9], axis=-1)
        block10 = self.decblock2(combinedBlock9)
        #print("UNET block 10 shape: ", block10.shape)
        combinedBlock10 = tf.concat([block6, block10], axis=-1)
        block11 = self.decblock3(combinedBlock10)
        #print("UNET block 11 shape: ", block11.shape)
        combinedBlock11 = tf.concat([block5, block11], axis=-1)
        block12 = self.decblock4(combinedBlock11)
        #print("UNET block 12 shape: ", block12.shape)
        combinedBlock12 = tf.concat([block4, block12], axis=-1)
        block13 = self.decblock5(combinedBlock12)
        #print("UNET block 13 shape: ", block13.shape)
        combinedBlock13 = tf.concat([block3, block13], axis=-1)
        block14 = self.decblock6(combinedBlock13)
        #print("UNET block 14 shape: ", block14.shape)
        combinedBlock14 = tf.concat([block2, block14], axis=-1)
        block15 = self.decblock7(combinedBlock14)
        #print("UNET block 15 shape: ", block15.shape)
        combinedBlock15 = tf.concat([block1, block15], axis=-1)
        block16 = self.decblock8(combinedBlock15)
        #print("UNET block 16 shape: ", block16.shape)

        #block16 is the output of the UNET, but we add a thing at the end of it as well. 
        
        return block16

