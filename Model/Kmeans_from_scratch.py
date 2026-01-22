import numpy as np
from Model.Base import BaseModel

class KmeansScratch(BaseModel):
    def __init__(self,k,max_iters=500):
        self.max_iters=max_iters
        self.k=k
    
    def init_centroids(self,X):
        randix=np.random.permutation(X.shape[0])
        self.centroids=X[randix[:self.k]]
        return self.centroids
        
    
    def find_centroid(self,X):
        # k=self.centroids.shape[0]
        idx = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            x_i=X[i]
            mini_dist=float('inf')
            for j in range(self.k):
                dist=np.sum((x_i-self.centroids[j])**2)
                if dist<mini_dist:
                    mini_dist=dist
                    idx[i]=j
        return idx
    
    def compute_centroid(self,X,idx):
        m,n=X.shape
        centroids_adjust=np.zeros((self.k,n))
        for i in range(self.k):
            points=X[idx==i]
            if len(points)==0:
                centroids_adjust[i]=X[np.random.randint(0,m)]
            else:
                centroids_adjust[i]=np.mean(points,axis=0)
        return centroids_adjust
    
    def compute_cost(self,X,idx):
         cost=0
         for i in range(self.k):
             points=X[idx==i]
             cost+=np.sum((points-self.centroids[i])**2)
            
         return cost
    def run_kmeans(self,X):
        m,n=X.shape
        self.init_centroids(X)
        idx=np.zeros(m)
        costs=[]
        for i in range(self.max_iters):
            idx=self.find_centroid(X)
            new_centroids=self.compute_centroid(X,idx)
            cost=self.compute_cost(X,idx)
            costs.append(cost)

            print(f"Cost for {i} iteration is:{cost}")
            if np.allclose(self.centroids,new_centroids):
                print(f"Converged at iteration {i}")
                break
            self.centroids=new_centroids

        return self.centroids,idx
    
    
    def fit(self, X, y=None):
        self.centroids,self.idx=self.run_kmeans(X)
        

    def predict(self, X):
        return self.find_centroid(X)

