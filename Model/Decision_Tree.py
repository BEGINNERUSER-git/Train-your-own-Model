from Model.Base  import BaseModel
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifierSklearn(BaseModel):
    def __init__(self,max_depth,min_sample_spilt,min_sample_leaf):
         self.model=DecisionTreeClassifier(
            self.max_depth,
            self.min_sample_spilt,
            self.min_sample_leaf
        )
    
    def fit(self, X, y):
         self.model.fit(X,y)
    
    def predict(self, X):
         return super().predict(X)

        