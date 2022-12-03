import tensorflow as tf
class TrainStepTime(tf.keras.metrics.Metric):
    def __init__(self, name):
        super().__init__(name)
        self.timeStart = 0
        self.timeStop = 0
        self.timeSpent = 0
        
    def update_state(self, timeStart, timeStop):
        self.timeStart = timeStart
        self.timeStop = timeStop
        self.timeSpent = timeStop - timeStart
        return

    def reset_state(self):
        self.timeStart = 0
        self.timeStop = 0 
        self.timeSpent = 0

    def result(self):
        return self.timeSpent