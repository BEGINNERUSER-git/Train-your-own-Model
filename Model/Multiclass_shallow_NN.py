from Model.Base import BaseModel
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
class MulticlassShallowNN(BaseModel):
    def __init__(self,max_iters=500,lr=0.001):
        self.max_iters = max_iters
        self.lr = lr
        self.input_shape = None
        self.num_classes = None
        self.model = Sequential(
            [
                Input(shape=(self.input_shape,)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(self.num_classes, activation='softmax')
            ]
        )
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]

        )
    def fit(self, X, y):
        self.input_shape = X.shape[1]
        self.num_classes = len(set(y))
        self.model.fit(X, y, epochs=self.max_iters, batch_size=32, verbose=0)
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return tf.argmax(predictions, axis=1)
    
