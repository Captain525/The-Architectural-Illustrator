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

        genPred = self.discriminator((generated, Y))
        realPred = self.discriminator((X, Y))
        
        return generated, genPred, realPred
    
    def batch_step(self, data, training):
        X = data[0]
        Y = data[1]
        batchSize = X.shape[0]
        with tf.GradientTape() as disTape: 
            #forward pass. 
            generated, predGen, predReal= self(data, training)
            discriminatorLoss = self.discriminator.compute_loss((generated, X), predGen, predReal)
            self.dLoss.update_state(discriminatorLoss)
        if(training):
            disGrad = disTape.gradient(discriminatorLoss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(disGrad, self.discriminator.trainable_variables))
        with tf.GradientTape() as genTape:
            generated, genPredict, predReal = self((None, Y), training)
            #is there a way to make this better? 
            assert(predReal is None)
            assert(genPredict.shape[0] == batchSize)
            generatorLoss = self.generator.compute_loss((generated, X), genPredict, predReal)
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

    @tf.function
    def train_step(self, data):
        
        return self.batch_step(data, True)
    @tf.function
    def test_step(self, data):
        return self.batch_step(data, False)

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