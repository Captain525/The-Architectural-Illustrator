import tensorflow as tf
import matplotlib.pyplot as plt
class displayImages(tf.keras.callbacks.Callback):
    """
    Callback designed to display the real images, the edge images, the generated images from teh edge images, 
    and the result of the patchGAN discriminator. 

    THis is what we used to get the images shown in the report and on our poster. 
    
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
        testImages = self.testData[0:numExamples]
        testEdges = self.testEdges[0:numExamples]
        generatedTrainImages, genPredictions, realPredictions= self.model((testImages, testEdges), training = False)
        trainEdgesTiled = tf.tile(testEdges, multiples = [1, 1, 1, 3])
        stackedImages = tf.stack([testImages, trainEdgesTiled, generatedTrainImages], axis=0)
        assert(stackedImages.shape == (3, numExamples, 256,256, 3))
        
        rows = numExamples
        columns = 5
        fig = plt.figure(figsize=(10, 10))
        
        for i in range(0, numExamples):
      
            for j in range(columns-2):
           
                index = i*columns + j + 1
                fig.add_subplot(rows, columns, index)
                plt.imshow(stackedImages[j, i, :], aspect = 'auto')  
                plt.axis('off')
                predProbability = 0
                if(j == 0):
                    predProbability = tf.reduce_mean(realPredictions[i])
                elif(j == 2):
                    predProbability = tf.reduce_mean(genPredictions[i])

           
        
            fig.add_subplot(rows,columns, i*columns + 4)
            plt.imshow(realPredictions[i], aspect = 'auto')
            plt.axis('off')
       
      
            fig.add_subplot(rows,columns, i*columns + 5)
            plt.imshow(genPredictions[i], aspect = 'auto')
            plt.axis('off')
         
        plt.subplots_adjust(hspace=0, wspace = 0)
        plt.show()
        print("done callback")