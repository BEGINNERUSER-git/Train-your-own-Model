from Model.Base import BaseModel
from sklearn.cluster import KMeans

class KmeansSklearn(BaseModel):
    def __init__(self, k, max_iters=500):
        
        self.k = k
        self.max_iters = max_iters
        self.model = KMeans(n_clusters=self.k, max_iter=self.max_iters)
    
    def fit(self, X, y=None):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
