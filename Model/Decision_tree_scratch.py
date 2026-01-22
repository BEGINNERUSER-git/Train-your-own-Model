from Model.Base import BaseModel
import numpy as np

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value    


class DecisionTreeScratch(BaseModel):
    def __init__(self,max_depth,min_samples_split,
    min_samples_leaf):
        self.max_depth=max_depth
        self.min_samples_spilt=min_samples_split
        self.min_samples_leaf=min_samples_leaf
    
    def entropy(self,p):
        if(p==0) or (p==1):
          return 0
        else:
            return -p*np.log2(p)-(1-p)*np.log2(1-p)
    
    def spilt_indices(self,X,index_feature,threshold):
        self.left_indices=[]
        self.right_indices=[]
        for i,x in enumerate(X):
            if x[index_feature]<=threshold:
                self.left_indices.append(i)
            else:
                self.right_indices.append(i)
        return self.left_indices,self.right_indices
    
    def weighted_entropy(self,X,y,left_,right_):
        w_left=len(left_)/len(X)
        w_right=len(right_)/len(X)
        p_1_left=self.entropy(sum(y[left_])/len(left_))
        p_1_right=self.entropy(sum(y[right_])/len(right_))
        return (w_left*p_1_left)+(w_right*p_1_right)
    def information_gain(self,X,y,left_,right_):
        p_root=self.entropy(sum(y)/len(y))
        w_entropy=self.weighted_entropy(X,y,left_,right_)
        return p_root-w_entropy
    
    def best_spilt(self,X,y):
        n=X.shape[1]
        self.best_threshold=0
        self.best_feature=None
        self.best_gain=0
        self.best_set=None
        for i in range(n):
            threshold=np.unique(X[:,i])
            for th in threshold:
                left_indices,right_indices=self.spilt_indices(X,i,th)
                if len(left_indices)==0 or len(right_indices)==0:
                    continue
                gain=self.information_gain(X,y,left_indices,right_indices)
                if  gain>self.best_gain:
                    self.best_threshold=th
                    self.best_feature=i
                    self.best_gain=gain
                    self.best_set=(left_indices,right_indices)
        return self.best_feature,self.best_threshold,self.best_set

    
    def build_tree(self,X,y,depth):
        if len(set(y))==1:
            return Node(value=y[0])
        
        if depth>=self.max_depth or len(y)<self.min_samples_spilt:
            return Node(value=int(np.round(np.mean(y))))
        feature,threshold,set=self.best_spilt(X,y) 
        if feature is None:
            return Node(value=int(np.round(np.mean(y))))     
        left_,right_=set

        
        left_child=self.build_tree(
            X[left_],y[left_],depth+1
        )
        right_child=self.build_tree(
            X[right_],y[right_],depth+1
        )
        return Node(feature,threshold,left_child,right_child)
    
    def fit(self,X,y):
        self.root=self.build_tree(X,y,depth=0)

    def predict_one(self,X,tree):
        if tree.value is not None:
            return tree.value

        if X[tree.feature]<=tree.threshold:
                return self.predict_one(X,tree.left)
        return self.predict_one(X,tree.right)
    
    def predict(self, X):
        return np.array([self.predict_one(x,self.root)] for x in X)
