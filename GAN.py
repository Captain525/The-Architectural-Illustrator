import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
class GAN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.dLoss = tf.keras.metrics.Mean(name = "dLoss")
        self.gLoss = tf.keras.metrics.Mean(name = "gLoss")
        self.sumLoss = tf.keras.metrics.Mean(name = "sumLoss")
        self.listMetrics = [self.dLoss, self.gLoss, self.sumLoss]
    def call(self, data, training):
        X = data[0]
        Y = data[1]
        
        batchSize, height, width, depth = X.shape
        assert(Y.shape[0] == batchSize)
        generated= self.generator(Y, training)
        assert(generated.shape == (batchSize, height, width, depth))
        
        #generator output shape: batchSize x generatedImageSiz
        data = tf.concat([X, generated], axis=0)

        pred = self.discriminator((data, Y), training)
        assert(pred.shape == (data.shape[0], 1))
        return generated, data, pred
    @tf.function
    def train_step(self, data):
 
        batchSize = data[0].shape
        predLabels, genLabels= self.generateLabels(batchSize, batchSize)
        with tf.GradientTape() as disTape: 
            #forward pass. 
            generated, dataCombined, pred = self(data, True)
            
            discriminatorLoss = self.discriminator.compute_loss(dataCombined, predLabels, pred)
            self.dLoss.update_state(discriminatorLoss)
        #calculate the gradients of each model's loss w.r.t the weights for each. 
        disGrad = disTape.gradient(discriminatorLoss, self.discriminator.trainable_variables)
        
        #update the gradients for both the generator and the discriminator
        self.discriminator.optimizer.apply_gradients(zip(disGrad, self.discriminator.trainable_variables))

        with tf.GradientTape() as genTape:
            generated, dataCombined, pred = self(data, True)
            #is there a way to make this better? 
            genPredict = pred[batchSize:]
            assert(genPredict.shape[0] == batchSize)
            generatorLoss = self.generator.compute_loss(generated, genLabels, genPredict)
            self.gLoss.update_state(generatorLoss)
        self.sumLoss.update_state(generatorLoss + discriminatorLoss)
        genGrad = genTape.gradient(generatorLoss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(genGrad, self.generator.trainable_variables))

        metricDict = self.evalMetrics()

        return  metricDict
    def compile(self, optimizerGen, optimizerDis, lossFxnGen, lossFxnDis):
        super().compile()
        self.generator.compile(optimizerGen, lossFxnGen)
        self.discriminator.compile(optimizerDis, lossFxnDis)
        #maybe add metrics here. 
    def generateLabels(self, genShape, predShape):
        """
        Generate labels for the loss function. We wish to use binary cross entropy. 

        For the discriminator, give the real images the label of 1, give the fake images the label of 0. We wish to maximize log(D(x)) + log(1-D(G(z)))
        which means the discriminator is MOST accurate. Binary crossentropy takes the opposite of this, so we're MINIMZING the NEGATIVE. 

        For the generator, give the "fake" images the label of 1(all generator ones have a label of 1, meaning we're CALLING them real even though they aren't). 
        Since we WANT them to be seen as real, those are the labels we give them. 
        This is equivalent to giving them labels of zero and minimizing log(1-D(G(z))). However, we didn't do that for 2 reasons. 
        1. Doesn't work well with binary crossentropy, since that takes the negative and we'd have to "un-take" the negative. 
        2. The GAN paper says to maximize log(D(G(z))) because it has stronger gradients earlier in learning. 
        """
        #make them a batch of size 2*batchSize. 
       
        #we're assumign the output is batchSize by 1. We might have to do something differently if we have 2 or more dimensions. 
        discrimLabels = tf.concat([tf.ones(genShape), tf.zeros(predShape)], axis=0)

        genLabels = tf.ones(genShape)
        return discrimLabels, genLabels
    def generateImages(self, Y):
        """
        Generate some images from the conditions. 
        """
        generated = self.generator(Y, training = False)
        return generated
    def evalMetrics(self):
        return {metric.n:metric.result() for metric in self.listMetrics}