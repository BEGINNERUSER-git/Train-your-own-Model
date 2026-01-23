from sklearn.linear_model import LinearRegression
from Model.Base import BaseModel

class LinearRegressionSklearn(BaseModel):
    def __init__(self):
        self.model=LinearRegression()
    def fit(self,X,y):
        self.model.fit(X,y)
    def predict(self, X):
        return super().predict(X)




