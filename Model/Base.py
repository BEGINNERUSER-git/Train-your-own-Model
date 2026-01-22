class BaseModel:
    def fit(self,X,y=None):
        raise NotImplementedError
    def predict(self,X):
        raise NotImplementedError
    
