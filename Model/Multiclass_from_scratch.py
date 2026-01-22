from Model.Base import BaseModel
import numpy as np

class MulticlassScratch(BaseModel):
    def __init__(self,lr=0.1,max_iters=500):
        super().__init__()
        self.alpha=lr
        self.max_iters=max_iters
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def relu(self,z):
        return np.maximum(0,z)
    
    def relu_derivative(self,z):
        return np.where(z>0,1,0)
    
    def softmax(self,z):
        exp_z=np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)

    def init_weights(self, input_dim, num_classes):
        np.random.seed(42)

        self.W1 = np.random.randn(input_dim, 128) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, 128))

        self.W2 = np.random.randn(128, 64) * np.sqrt(2 / 128)
        self.b2 = np.zeros((1, 64))

        self.W3 = np.random.randn(64, num_classes) * np.sqrt(2 / 64)
        self.b3 = np.zeros((1, num_classes))

    def forward(self,X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)

        cache=(self.Z1,self.A1,self.Z2,self.A2,self.Z3,self.A3)
        return self.A3,cache
    
    def compute_loss(self,Y_true,Y_pred):
        n = Y_true.shape[0]
        return -np.mean(np.log(Y_pred[np.arange(n), Y_true] + 1e-9))

    def backward(self,X,Y_true,cache):
        m = X.shape[0]
        Z1,A1,Z2,A2,Z3,A3 = cache

        dZ3 = A3
        dZ3[np.arange(m), Y_true] -= 1
        dZ3 /= m

        dW3 =   A2.T@dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3@self.W3.T
        dZ2 = dA2 * self.relu_derivative(Z2)

        dW2 = A1.T@dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2@self.W2.T
        dZ1 = dA1 * self.relu_derivative(Z1)

        dW1 = X.T@dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W3 -= self.alpha * dW3
        self.b3 -= self.alpha * db3
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
    def fit(self,X,Y):
        num_classes=len(np.unique(Y))
        self.init_weights(X.shape[1],num_classes)
        for i in range(self.max_iters):
            Y_pred,cache=self.forward(X)
            loss=self.compute_loss(Y,Y_pred)
            self.backward(X,Y,cache)
            if i%100==0:
                print(f"Iteration {i}, loss: {loss:.4f}")
    def predict(self,X):
        Y_pred, _ = self.forward(X)
        return np.argmax(Y_pred, axis=1)
    

