import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fill_values = {}
        self.dummy_columns = None
        self.cat_cols = None
        
    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows from the DataFrame."""
        self.data = self.data.drop_duplicates()
        return self.data
    def fill_missing_values(self, data_num, strategy: str = 'mean') -> pd.DataFrame:
        for col in data_num:
            if strategy == 'mean':
                value = self.data[col].mean()
            elif strategy == 'median':
                value = self.data[col].median()
            elif strategy == 'mode':
                value = self.data[col].mode().iloc[0]
            else:
                raise ValueError("Invalid strategy")

            self.fill_values[col] = value
            self.data[col] = self.data[col].fillna(value)

        return self.data

    def encoding_coategorical(self,cat_col)->pd.DataFrame:
        """Encode categorical variables using one-hot encoding."""
        self.cat_cols = cat_col
        if isinstance(cat_col, list) and 'date' in cat_col:
            self.data['year'] = self.data['date'].dt.year
            self.data['month'] = self.data['date'].dt.month
            self.data['day'] = self.data['date'].dt.day
            self.data['dayofweek'] = self.data['date'].dt.dayofweek

        else:
           self.data = pd.get_dummies(self.data, columns=cat_col, drop_first=True).astype('float64')
        
           self.dummy_columns =self.data.columns.tolist()

        return self.data
    
    def drop_columns(self,id_columns:list)->pd.DataFrame:
        """Drop specified ID columns from the DataFrame."""
        self.data = self.data.drop(columns=id_columns)
        return self.data
    def data_clipped_str(self)->pd.DataFrame:
        """Trim whitespace and retrieving valuable information  from string columns."""
        str_cols = self.data.select_dtypes(include=['object']).columns
        for col in str_cols:
            self.data[col] = self.data[col].str.strip()
        return self.data
    def custom_column(self, col,func)->pd.DataFrame:
        """Apply a custom function to a specific column."""
        if func=='Log Transformation':
            self.data[col]=np.log1p(self.data[col])
        elif func=='Square Root Transformation':
            self.data[col]=np.sqrt(self.data[col])
        elif func=='Square Transformation':
            self.data[col]=np.square(self.data[col])
        elif func=='Absolute Transformation':
            self.data[col]=np.abs(self.data[col])   
        return self.data
    
    def transform(self, new_data: pd.DataFrame) -> pd.DataFrame:
        df = new_data.copy()

        for col, value in self.fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        if self.cat_cols:
            df = pd.get_dummies(df, columns=self.cat_cols, drop_first=True)
            for col in self.dummy_columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[self.dummy_columns]

        return df

    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        return self.data
    
