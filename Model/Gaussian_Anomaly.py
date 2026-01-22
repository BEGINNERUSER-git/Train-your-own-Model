import numpy as np
import streamlit as st
class AnomalyDetection():
    def __init__(self):
        super().__init__()

    def estimate_gaussian(self,X):
        m,n=X.shape
        self.mean=np.mean(X,axis=0)
        self.var=np.var(X,axis=0)
        self.var[self.var == 0] = 1e-9
        return self.mean,self.var
    
    def multivariate_gaussian(self,X):
        n=len(self.mean)
        p=(1/np.sqrt((2*np.pi)**n*np.prod(self.var)))*np.exp((-0.5)*np.sum(((X-self.mean)**2)/self.var,axis=1))
        return p

    def select_threshold(self,y_true,y_pred):
        self.best_f1=0
        f1=0
        self.best_e=0
        step_size=(max(y_pred)-min(y_pred))/1000
        for epsilon in np.arange(min(y_pred),max(y_pred),step_size):
            pred=y_pred<epsilon
            tp=np.sum((pred==1) & (y_true==1))
            fp=np.sum((y_true==0) &(pred==1))
            fn=np.sum((y_true==1) &(pred==0))
            if tp + fp == 0 or tp + fn == 0:
                continue
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            f1=(2*precision*recall)/(precision+recall)

            if f1>self.best_f1:
                self.best_f1=f1
                self.best_e=epsilon
        return self.best_e,self.best_f1
    
    def fit(self, X_train, y_train,X_cv,y_cv):
        X_normal = X_train[y_train == 0]
        mean,var=self.estimate_gaussian(X_normal)
        p=self.multivariate_gaussian(X_cv)
        self.epsilon,self.f1_Score=self.select_threshold(y_cv,p)
    
    def predict(self,X):
        p=self.multivariate_gaussian(X)
        return np.where(p<self.epsilon,1,0)




        
