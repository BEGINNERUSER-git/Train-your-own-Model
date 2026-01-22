import numpy as np
from Model.Base import BaseModel

class BinaryClassificationScratch(BaseModel):

    def __init__(self, lr, num_iters,threshold):
        self.lr = lr
        self.num_iters = num_iters
        self.threshold = threshold

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_class_weights(self, y):
        N = len(y)
        pos = np.sum(y)
        neg = N - pos
        self.pos_weight = N / (2 * pos)
        self.neg_weight = N / (2 * neg)

    def loss(self, X, y):
        z = np.dot(X, self.w) + self.b
        y_hat = self.sigmoid(z)

        eps = 1e-15
        loss = (
            - self.pos_weight * y * np.log(y_hat + eps)
            - self.neg_weight * (1 - y) * np.log(1 - y_hat + eps)
        )

        return np.mean(loss)

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        self.compute_class_weights(y)

        for _ in range(self.num_iters):
            z = np.dot(X, self.w) + self.b
            y_hat = self.sigmoid(z)

            error = y_hat - y
            weights = np.where(y == 1, self.pos_weight, self.neg_weight)

            dw = (1 / m) * np.dot(X.T, weights * error)
            db = (1 / m) * np.sum(weights * error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        probs = self.sigmoid(z)
        return (probs >= self.threshold).astype(int)