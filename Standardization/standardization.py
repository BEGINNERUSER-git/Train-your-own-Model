from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class Standardization:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaled_data = None

    def get_continuous_columns(self, data: pd.DataFrame):
        """Return a list of continuous (numerical) columns in the DataFrame."""
        numeric_col= data.select_dtypes(include=[np.number]).columns
        continuous_columns = [col for col in numeric_col if data[col].nunique() > 2]
        return continuous_columns

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler to the data and transform it."""
        self.scaled_data = self.get_continuous_columns(data)
        if not self.scaled_data:
            raise ValueError("No continuous columns found for standardization.")
        data_scaled = data.copy()
        data_scaled[self.scaled_data] = self.scaler.fit_transform(data[self.scaled_data])
        return pd.DataFrame(data_scaled, columns=data.columns)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the already fitted scaler."""
        if self.scaled_data is None:
            raise ValueError("Scaler has not been fitted yet. Please call fit_transform first.")
        data_scaled = data.copy()
        data_scaled[self.scaled_data] = self.scaler.transform(data[self.scaled_data])
        return pd.DataFrame(data_scaled, columns=data.columns)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform the scaled data back to original scale."""
        if self.scaled_data is None:
            raise ValueError("Scaler has not been fitted yet. Please call fit_transform first.")    
        data_scaled = data.copy()
        data_scaled[self.scaled_data]  = self.scaler.inverse_transform(data[self.scaled_data])
        return pd.DataFrame(data_scaled, columns=data.columns)        
    
    