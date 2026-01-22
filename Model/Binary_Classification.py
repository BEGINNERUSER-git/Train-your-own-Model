from Model.Base import BaseModel
from sklearn.linear_model import LogisticRegression

class BinaryClassificationSklearn(BaseModel):
    def __init__(self):
        self.model=LogisticRegression(class_weight='balanced')
    def fit(self,X,y):
        return self.model.fit(X,y)
    def predict(self,X):
        return self.model.predict(X)
    