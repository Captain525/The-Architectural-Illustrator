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
        #generate data
        generated = self.generator(Y, training) 
        #run discriminator on generated examples. 
        genPred = self.discriminator((generated, Y))
        #in case where want gradients of generator, don't need this. 
        realPred = None
        #in case where you're calculating gradients of discriminator, run the real predictions. 
        if(X is not None):
            #run discriminator on real examples. 
            realPred = self.discriminator((X, Y))
        #return generated examples, predictions of generated examples, and predictions of real examples
        return generated, genPred, realPred
    
    def batch_step(self, data, training):
        """
        Called from both train step and test step, makes the methods simpler by keeping all the code in one place. 

        X - real examples of images. 
        Y - outlines of those specific real examples. This is the conditional input. 

        """
        X = data[0]
        Y = data[1]
        #print("batch step eager? :",tf.executing_eagerly())
        #calculate discriminator gradients first. 
        with tf.GradientTape() as disTape: 
            #forward pass. 
            generated, predGen, predReal= self(data, training)
            discriminatorLoss = self.discriminator.compute_loss((generated, X), predGen, predReal)
            self.dLoss.update_state(discriminatorLoss)
        if(training):
            #if training, calculate gradients and update weights. 
            disGrad = disTape.gradient(discriminatorLoss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(disGrad, self.discriminator.trainable_variables))
        #now once those are updated, calculate generator weights. 
        with tf.GradientTape() as genTape:
            #none indicates to the call function we're working with the generator. 
            generated, genPredict, predReal = self((None, Y), training)
            #is there a way to make this better? 
            assert(predReal is None)
            #calculate the loss of the generator. Don't need real predictions. 
            generatorLoss = self.generator.compute_loss((generated, X), genPredict, predReal)
            self.gLoss.update_state(generatorLoss)
        if(training):
            print("about to update gradients")
            genGrad = genTape.gradient(generatorLoss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(genGrad, self.generator.trainable_variables))
        self.updateStates(not training, generatorLoss, discriminatorLoss)
        print("end of batch step")
        return self.evalMetrics(training)

    def compile(self, optimizerGen, optimizerDis, lossFxnGen, lossFxnDis, metrics = None, steps_per_execution = 1):

        super().compile(steps_per_execution = steps_per_execution, metrics = metrics)
        self.generator.compile(optimizerGen, lossFxnGen)
        self.discriminator.compile(optimizerDis, lossFxnDis)
        #maybe add metrics here. 
        #self.createMetrics()

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
    
        self.listMetricsTrain = [self.dLoss, self.gLoss, self.sumLoss]
        self.listMetricsTest = [self.dValLoss, self.gValLoss, self.valSumLoss]
        self.listMetrics = self.listMetricsTrain + self.listMetricsTest
        return self.listMetrics
    def updateStates(self, val, gLoss, dLoss):
        if val:
            self.dValLoss.update_state(dLoss)
            self.gValLoss.update_state(gLoss)
            self.valSumLoss.update_state(dLoss+gLoss)

        else:
            self.dLoss.update_state(dLoss)
            self.gLoss.update_state(gLoss)
            self.sumLoss.update_state(dLoss + gLoss)
    def resetStates(self):
        for metric in self.listMetrics:
            metric.reset_state()
    def evalMetrics(self, training):
        if training:
            return self.evalMetricsTrain()
        else:
            return self.evalMetricsTest()
    def evalMetricsTest(self):
        return {metric.name:metric.result() for metric in self.listMetricsTest}
    def evalMetricsTrain(self):
        return {metric.name:metric.result() for metric in self.listMetricsTrain}