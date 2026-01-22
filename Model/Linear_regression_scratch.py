from Model.Base import BaseModel
import numpy as np  
class LinearRegressionScratch(BaseModel):
    def __init__(self,lr=0.01,max_iters=1000):
        self.lr=lr
        self.max_iters=max_iters

    def cost(self,X,y):
        m=X.shape[0]
        y_pred=np.dot(X,self.w)+self.b
        cost=(1/(2*m))*np.sum((y_pred - y)**2)
        return cost
    
    def partial_derivative(self,X,y):
        f_wb=np.dot(X,self.w)+self.b
        m,n=X.shape
        dj_dw=(1/m)*(np.dot(X.T,(f_wb - y)))
        dj_db=(1/m)*np.sum(f_wb - y)
        return dj_dw,dj_db
    
    def gradient_descent(self,X,y):
        m,n=X.shape
        j_history=[]
        for i in range(self.max_iters):
            dj_dw,dj_db=self.partial_derivative(X,y)
            self.w=self.w - self.lr*dj_dw
            self.b=self.b - self.lr*dj_db
            if i%100==0:
                j_history.append(self.cost(X,y))
            #check id not convergence break
                        
        return self.w,self.b

    def fit(self,X,y):
        m,n=X.shape
        self.w=np.zeros(n)
        self.b=0
        self.w,self.b=self.gradient_descent(X,y)


    def predict(self,X):
        return X.dot(self.w)+self.b
    
