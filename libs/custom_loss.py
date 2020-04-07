import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    
    def __init__(self, 
                 gamma=0.9):
        super(FocalLoss, self).__init__()

        self.gamma = gamma


    def call(self, y_true, y_pred):
        return loss        

class ClassBalancedLoss(tf.keras.losses.Loss):
    
    def __init__(self, 
                 gamma=0.9):
        super(ClassBalancedLoss, self).__init__()

        self.gamma = gamma


    def call(self, y_true, y_pred):
        return loss        

class MaxMarginLoss(tf.keras.losses.Loss):
    
    def __init__(self, 
                 gamma=0.9):
        super(MaxMarginLoss, self).__init__()

        self.gamma = gamma


    def call(self, y_true, y_pred):
        return loss        
