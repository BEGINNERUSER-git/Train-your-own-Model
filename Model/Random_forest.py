from Model.Base import BaseModel
from sklearn.ensemble import RandomForestClassifier
class RandomForestClassifierSklearn(BaseModel):
    def __init__(self,n_estimators=100,max_depth=None,min_samples_split=2,min_samples_leaf=1):
        super().__init__()
        self.model=RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    
    def fit(self, X, y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    