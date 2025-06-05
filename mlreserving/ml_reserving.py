"""
Main implementation of machine learning reserving
"""

import pandas as pd
import numpy as np
from collections import namedtuple
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from nnetsauce import PredictionInterval

class MLReserving:
    """
    Machine learning based reserving model
    
    Parameters
    ----------
    model : object, optional
        model to use (must implement fit and predict methods), default is RandomForestRegressor
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
            model = RandomForestRegressor(random_state=random_state)
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
        
        full_data["to_predict"] = full_data[value_col].isna()

        self.full_data_ = deepcopy(full_data)
        self.full_data_lower_ = deepcopy(full_data)
        self.full_data_upper_ = deepcopy(full_data)
        
        train_data = full_data[~full_data["to_predict"]]
        test_data = full_data[full_data["to_predict"]]
        
        X_train = train_data[[origin_col, "dev", "calendar"]].values
        y_train = train_data[value_col].values
        self.X_test_ = test_data[[origin_col, "dev", "calendar"]].values

        self.model.fit(X_train, y_train)
        
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

            self.full_data_.loc[to_predict, self.value_col] = preds.mean
            self.full_data_lower_.loc[to_predict, self.value_col] = preds.lower
            self.full_data_upper_.loc[to_predict, self.value_col] = preds.upper

            DescribeResult = namedtuple("DescribeResult", ("mean", "lower", "upper"))
            return DescribeResult(self.full_data_.pivot(index=self.origin_col, 
                                                        columns="dev", 
                                                        values=self.value_col).sort_index().T, 
                                  self.full_data_lower_.pivot(index=self.origin_col, 
                                                        columns="dev", 
                                                        values=self.value_col).sort_index().T, 
                                  self.full_data_upper_.pivot(index=self.origin_col, 
                                                        columns="dev", 
                                                        values=self.value_col).sort_index().T)
        else: 

            raise NotImplementedError