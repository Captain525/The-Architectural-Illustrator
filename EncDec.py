import tensorflow as tf
from Blocks import ConvBlock, ConvTBlock
class EncDec(tf.keras.Layer):

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
        self.decblock2 = ConvTBlock(512, True, True)
        self.decblock3 = ConvTBlock(512, True, True)
        self.decblock4 = ConvTBlock(512, True, False)
        self.decblock5 = ConvTBlock(256, True, False)
        self.decblock6 = ConvTBlock(128, True, False)
   
        self.decblock7 = ConvTBlock(64, True, False)

    def call(self, input):
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
      
        block10 = self.decblock2(block9)
      
        block11 = self.decblock3(block10)

        block12 = self.decblock4(block11)
    
        block13 = self.decblock5(block12)
       
        block14 = self.decblock6(block13)
      
        block15 = self.decblock7(block14)

        return block15