"""
Main implementation of machine learning reserving
"""

import pandas as pd
import numpy as np
from collections import namedtuple
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from nnetsauce import PredictionInterval

def arcsinh(x):
    """Arcsinh transformation with offset for zero values"""
    return np.arcsinh(x + 1)

def inv_arcsinh(x):
    """Inverse arcsinh transformation with offset"""
    return np.sinh(x) - 1

class MLReserving:
    """
    Machine learning based reserving model
    
    Parameters
    ----------
    model : object, optional
        model to use (must implement fit and predict methods), default is RidgeCV
    level: a float;
        Confidence level for prediction intervals. Default is 95,
        equivalent to a miscoverage error of 5 (%)
    replications: an integer;
        Number of replications for simulated conformal (default is `None`),
        for type_pi = "bootstrap" or "kde"
    type_pi: a string;
        type of prediction interval: currently `None`
        split conformal prediction without simulation, "kde" or "bootstrap"
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, 
                 model=None, 
                 level=95,
                 replications=None,
                 type_pi=None,
                 random_state=42):
        if model is None:
            model = RidgeCV(alphas=[10**i for i in range(-5, 5)])
        self.model = PredictionInterval(model, level=level, 
                                        type_pi=type_pi, 
                                        type_split="sequential",
                                        replications=replications)
        self.level = level 
        self.replications = replications
        self.type_pi = type_pi 
        self.origin_col = None
        self.development_col = None
        self.value_col = None
        self.max_dev = None
        self.origin_years = None
        self.X_test_ = None 
        self.full_data_ = None 
        self.full_data_upper_ = None 
        self.full_data_lower_ = None 
        self.full_data_sims_ = []
        self.scaler = StandardScaler()
        
    def fit(self, data, origin_col="origin", 
            development_col="development", 
            value_col="values"):
        """
        Fit the model to the triangle data
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data with origin, development, and value columns
        origin_col : str, default="origin"
            Name of the origin year column
        development_col : str, default="development"
            Name of the development year column
        value_col : str, default="values"
            Name of the value column
            
        Returns
        -------
        self : object
            Returns self
        """
        # Store column names
        self.origin_col = origin_col
        self.development_col = development_col
        self.value_col = value_col
        
        df = data.copy()
                
        df["dev"] = df[development_col] - df[origin_col]
        
        self.max_dev = df["dev"].max()
        self.origin_years = df[origin_col].unique()
        
        full_grid = pd.MultiIndex.from_product(
            [self.origin_years, range(self.max_dev)],
            names=[origin_col, "dev"]
        ).to_frame(index=False)
        
        full_data = pd.merge(
            full_grid, 
            df[[origin_col, "dev", value_col]], 
            on=[origin_col, "dev"], 
            how="left"
        )
        
        full_data["calendar"] = full_data[origin_col] + full_data["dev"]
        
        # Apply transformations
        full_data["log_origin"] = np.log(full_data[origin_col])
        full_data["log_dev"] = np.log(full_data["dev"] + 1)  # +1 to handle dev=0
        full_data["log_calendar"] = np.log(full_data["calendar"])
        
        # Transform response if not NaN
        full_data[f"arcsinh_{value_col}"] = full_data[value_col].apply(
            lambda x: arcsinh(x) if pd.notnull(x) else x
        )
        
        full_data["to_predict"] = full_data[value_col].isna()

        self.full_data_ = deepcopy(full_data)
        self.full_data_lower_ = deepcopy(full_data)
        self.full_data_upper_ = deepcopy(full_data)
        
        train_data = full_data[~full_data["to_predict"]]
        test_data = full_data[full_data["to_predict"]]
        
        # Prepare features for training
        feature_cols = ["log_origin", "log_dev", "log_calendar"]
        X_train = train_data[feature_cols].values
        X_test = test_data[feature_cols].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_ = self.scaler.transform(X_test)
        
        y_train = train_data[f"arcsinh_{value_col}"].values

        self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self):
        """
        Make predictions for the missing values in the triangle
        
        Parameters
        ----------
        level : int
            level of confidence
            
        Returns
        -------
        pandas.DataFrame
            Complete reserving triangle with predictions
        """
        preds = self.model.predict(self.X_test_, return_pi=True)            
        
        to_predict = self.full_data_["to_predict"]        

        if self.type_pi is None: 
            # Transform predictions back to original scale
            mean_pred = inv_arcsinh(preds.mean)
            lower_pred = inv_arcsinh(preds.lower)
            upper_pred = inv_arcsinh(preds.upper)
            
            # Ensure predictions are non-negative
            mean_pred = np.maximum(mean_pred, 0)
            lower_pred = np.maximum(lower_pred, 0)
            upper_pred = np.maximum(upper_pred, 0)
            
            # Store predictions in the full data
            self.full_data_.loc[to_predict, self.value_col] = mean_pred
            self.full_data_lower_.loc[to_predict, self.value_col] = lower_pred
            self.full_data_upper_.loc[to_predict, self.value_col] = upper_pred

            # Calculate triangles
            mean_triangle = self.full_data_.pivot(index=self.origin_col, 
                                                columns="dev", 
                                                values=self.value_col).sort_index().T
            lower_triangle = self.full_data_lower_.pivot(index=self.origin_col, 
                                                       columns="dev", 
                                                       values=self.value_col).sort_index().T
            upper_triangle = self.full_data_upper_.pivot(index=self.origin_col, 
                                                       columns="dev", 
                                                       values=self.value_col).sort_index().T

            # Calculate IBNR based on predicted values
            test_data = self.full_data_[to_predict]
            ibnr_mean = test_data.groupby(self.origin_col)[self.value_col].sum()
            ibnr_lower = self.full_data_lower_[to_predict].groupby(self.origin_col)[self.value_col].sum()
            ibnr_upper = self.full_data_upper_[to_predict].groupby(self.origin_col)[self.value_col].sum()

            DescribeResult = namedtuple("DescribeResult", 
                                      ("mean", "lower", "upper", "ibnr_mean", "ibnr_lower", "ibnr_upper"))
            return DescribeResult(mean_triangle, lower_triangle, upper_triangle,
                                ibnr_mean, ibnr_lower, ibnr_upper)
        else: 
            raise NotImplementedError