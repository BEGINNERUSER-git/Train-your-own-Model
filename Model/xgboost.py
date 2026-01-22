from Model.Base import BaseModel
import xgboost as xgb
class XGBoostClassifierSklearn(BaseModel):
    def __init__(self,n_estimators=100,learning_rate=0.1,max_depth=3):
        super().__init__()
        self.model=xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            objective='binary:logistic',
            eval_metric='logloss'
        )
    
    def fit(self, X, y):
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)