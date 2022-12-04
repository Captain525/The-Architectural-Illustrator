import tensorflow as tf
import matplotlib.pyplot as plt
class displayImages(tf.keras.callbacks.Callback):
    """
    Callback designed to display the real images, the edge images, the generated images from teh edge images, 
    and the result of the patchGAN discriminator. 
    
    """
    def __init__(self, trainData,trainEdges, testData, testEdges):
        self.trainData = trainData
        self.trainEdges = trainEdges
        self.testData = testData
        self.testEdges = testEdges

    def on_train_batch_end(self, batch, logs):
        return
       
    def on_epoch_end(self, batch, logs):
        self.displayImages(batch, logs)
        return 
    def displayImages(self, batch, logs):
        print("in callback")
        numExamples = 5
        trainImages = self.trainData[0:numExamples]
        trainEdges = self.trainEdges[0:numExamples]
        generatedTrainImages, genPredictions, realPredictions= self.model((trainImages, trainEdges), training = False)
        #generatedTestImages, genTestPred, realTestpred = self.model((self.testData[0:5], self.testEdges[0:5]), training =False)
        trainEdgesTiled = tf.tile(trainEdges, multiples = [1, 1, 1, 3])
        stackedImages = tf.stack([trainImages, trainEdgesTiled, generatedTrainImages], axis=0)
        assert(stackedImages.shape == (3, numExamples, 256,256, 3))
        #print("image types: ", generatedTrainImages)
        #print("max: ", tf.math.reduce_max(generatedTrainImages))
        #print("min: ", tf.math.reduce_min(generatedTrainImages))
        rows = numExamples
        columns = 5
        fig = plt.figure(figsize=(10, 10))
        #print("start iteration")
        for i in range(0, numExamples):
            #print("on i: ", i)
            for j in range(columns-2):
                #print("on j ", j)
                index = i*columns + j + 1
                fig.add_subplot(rows, columns, index)
                plt.imshow(stackedImages[j, i, :], aspect = 'auto')  
                plt.axis('off')
                predProbability = 0
                if(j == 0):
                    predProbability = tf.reduce_mean(realPredictions[i])
                elif(j == 2):
                    predProbability = tf.reduce_mean(genPredictions[i])

           
            #print("on j " , 3)
            fig.add_subplot(rows,columns, i*columns + 4)
            plt.imshow(realPredictions[i], aspect = 'auto')
            plt.axis('off')
       
            #print("on j ", 4)
            fig.add_subplot(rows,columns, i*columns + 5)
            plt.imshow(genPredictions[i], aspect = 'auto')
            plt.axis('off')
         
        plt.subplots_adjust(hspace=0, wspace = 0)
        plt.show()
        print("done callback")