import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
import time
import sys
from CustomMetrics import *
class GAN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        reg_coeff = 100
        self.generator = Generator(reg_coeff = reg_coeff)
        self.discriminator = Discriminator()

       
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
       
        
        return dataVal, pred
    
    def batch_step(self, data, training):
        X = data[0]
        Y = data[1]
        batchSize = X.shape[0]
        with tf.GradientTape() as disTape: 
            #forward pass. 
            dataCombined, predGen, predReal = self(data, training)
            discrimLabels, genLabels = self.genLabelsNew(predReal, predGen)

            discriminatorLoss = self.discriminator.compute_loss(dataCombined, discrimLabels, pred)
            self.dLoss.update_state(discriminatorLoss)
        if(training):
            disGrad = disTape.gradient(discriminatorLoss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(disGrad, self.discriminator.trainable_variables))
        with tf.GradientTape() as genTape:
            generated, genPredict = self((None, Y), training)
            #is there a way to make this better? 
            
            assert(genPredict.shape[0] == batchSize)
            generatorLoss = self.generator.compute_loss(generated, genLabels, genPredict)
            self.gLoss.update_state(generatorLoss)
        if(training):
            genGrad = genTape.gradient(generatorLoss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(genGrad, self.generator.trainable_variables))
        self.updateStates(not training, generatorLoss, discriminatorLoss)
        return self.evalMetricsTest()

    def compile(self, optimizerGen, optimizerDis, lossFxnGen, lossFxnDis):

        super().compile()
        self.generator.compile(optimizerGen, lossFxnGen)
        self.discriminator.compile(optimizerDis, lossFxnDis)
        #maybe add metrics here. 
        self.createMetrics()

    def genLabelsNew(self, discriminatorOutput):
        """
        discriminator output is size 2*batchSize shape. 
        """
        halfShape = int(discriminatorOutput.shape[0]/2)
        zeroBoolArr = tf.cast(0*discriminatorOutput, bool)
        oneBoolArr = tf.logical_not(zeroBoolArr)
        assert(discriminatorOutput.shape == zeroBoolArr.shape)
        rightLabels = tf.cast(oneBoolArr, dtype = tf.int32)
        wrongLabels = tf.cast(zeroBoolArr, dtype = tf.int32)

        discrimLabels = tf.concat([rightLabels, wrongLabels], axis=0)
        genLabels = rightLabels
        return discrimLabels, genLabels
    @tf.function
    def train_step(self, data):
        
        return self.batch_step(data, True)
    @tf.function
    def test_step(self, data):
        return self.batch_step(data, False)

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
    def createMetrics(self):
        self.dLoss = tf.keras.metrics.Mean(name = "dLoss")
        self.gLoss = tf.keras.metrics.Mean(name = "gLoss")
        self.sumLoss = tf.keras.metrics.Mean(name = "sumLoss")
        self.dValLoss = tf.keras.metrics.Mean(name = "valDLoss")
        self.gValLoss = tf.keras.metrics.Mean(name = "valGLoss")
        self.valSumLoss = tf.keras.metrics.Mean(name = "valSumLoss")
        self.trainStepTime = TrainStepTime(name = "tTime")
        self.testStepTime = TrainStepTime(name = "vTime")
        self.listMetricsTrain = [self.dLoss, self.gLoss, self.sumLoss, self.trainStepTime]
        self.listMetricsTest = [self.dValLoss, self.gValLoss, self.valSumLoss, self.testStepTime]
        self.listMetrics = self.listMetricsTrain + self.listMetricsTest
        return self.listMetrics
    def updateStates(self, val, gLoss, dLoss, timeStart=None, timeStop=None):
        if val:
            self.dValLoss.update_state(dLoss)
            self.gValLoss.update_state(gLoss)
            self.valSumLoss.update_state(dLoss+gLoss)
            if timeStart is not None and timeStop is not None:
                self.testStepTime.update_state(timeStart, timeStop)
        else:
            self.dLoss.update_state(dLoss)
            self.gLoss.update_state(gLoss)
            self.sumLoss.update_state(dLoss + gLoss)
            if timeStart is not None and timeStop is not None:
                self.trainStepTime.update_state(timeStart,timeStop)
    def resetStates(self):
        for metric in self.listMetrics:
            metric.reset_state()
    def evalMetricsTest(self):
        return {metric.name:metric.result() for metric in self.listMetricsTest}
    def evalMetricsTrain(self):
        return {metric.name:metric.result() for metric in self.listMetricsTrain}