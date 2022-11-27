import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
class GAN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.generator = Generator(.0001)
        self.discriminator = Discriminator()

        self.dLoss = tf.keras.metrics.Mean(name = "dLoss")
        self.gLoss = tf.keras.metrics.Mean(name = "gLoss")
        self.sumLoss = tf.keras.metrics.Mean(name = "sumLoss")
        self.listMetrics = [self.dLoss, self.gLoss, self.sumLoss]
    def call(self, data, training):
        X = data[0]
        Y = data[1]
        batchSize, height, width = Y.shape[0:3]
        generated = self.generator(Y, training)
        if X is None:
            #call from generator. 
            dataVal = generated
        else:
            #call from discrminator. 
            depth = X.shape[3]
            assert(generated.shape == (batchSize, height, width, depth))
        
            #generator output shape: batchSize x generatedImageSiz
            dataVal = tf.concat([X, generated], axis=0)
            #need to append Y to both X and generated, so need to double its size. MAKE SURE THIS LINES UP. 
            Y = tf.tile(Y, (2, 1, 1, 1))

        pred = self.discriminator((dataVal, Y), training)
        assert(pred.shape == (dataVal.shape[0], 1))
        return dataVal, pred
    #@tf.function
    def train_step(self, data):
        print("starting train step")
        X = data[0]
        Y = data[1]
        batchSize = X.shape[0]
        predLabels, genLabels= self.generateLabels(X)
        with tf.GradientTape() as disTape: 
            #forward pass. 
            dataCombined, pred = self(data, True)
            
            discriminatorLoss = self.discriminator.compute_loss(dataCombined, predLabels, pred)
            self.dLoss.update_state(discriminatorLoss)
        #calculate the gradients of each model's loss w.r.t the weights for each. 
        disGrad = disTape.gradient(discriminatorLoss, self.discriminator.trainable_variables)
        
        #update the gradients for both the generator and the discriminator
        self.discriminator.optimizer.apply_gradients(zip(disGrad, self.discriminator.trainable_variables))

        with tf.GradientTape() as genTape:
            generated, genPredict = self((None, Y), True)
            #is there a way to make this better? 
            
            assert(genPredict.shape[0] == batchSize)
            generatorLoss = self.generator.compute_loss(generated, genLabels, genPredict)
            self.gLoss.update_state(generatorLoss)
        self.sumLoss.update_state(generatorLoss + discriminatorLoss)
        genGrad = genTape.gradient(generatorLoss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(genGrad, self.generator.trainable_variables))

        metricDict = self.evalMetrics()
        print("finished train step")
        return  metricDict
    @tf.function
    def test_step(self, data):
        X = data[0]
        Y = data[1]
        batchSize = X.shape[0]
        predLabels, genLabels= self.generateLabels(X)
        dataCombined, pred = self(data, True)
        print("finished test step call")
        generated = dataCombined[batchSize:]
        predGen = pred[batchSize:]
        discriminatorLoss = self.discriminator.compute_loss(dataCombined, predLabels, pred)
        self.dLoss.update_state(discriminatorLoss)
        generatorLoss = self.generator.compute_loss(generated, genLabels, predGen)
        self.gLoss.update_state(generatorLoss)
        print("finished test step")
        return self.evalMetrics()

    def compile(self, optimizerGen, optimizerDis, lossFxnGen, lossFxnDis):
        super().compile()
        self.generator.compile(optimizerGen, lossFxnGen)
        self.discriminator.compile(optimizerDis, lossFxnDis)
        #maybe add metrics here. 
    def generateLabels(self, X):
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
        zeroBoolArr = tf.reduce_all(tf.cast(0*X, bool), axis=[1,2,3])
        oneBoolArr = tf.logical_not(zeroBoolArr)
        rightLabels = tf.cast(oneBoolArr, dtype=tf.int32)
        wrongLabels = tf.cast(zeroBoolArr, dtype = tf.int32)
        #we're assumign the output is batchSize by 1. We might have to do something differently if we have 2 or more dimensions. 
        discrimLabels = tf.concat([rightLabels, wrongLabels], axis=0)

        genLabels = rightLabels
        return discrimLabels, genLabels
    def generateImages(self, Y):
        """
        Generate some images from the conditions. 
        """
        generated = self.generator(Y, training = False)
        return generated
    def evalMetrics(self):
        return {metric.name:metric.result() for metric in self.listMetrics}