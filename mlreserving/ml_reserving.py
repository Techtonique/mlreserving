"""
Main implementation of machine learning reserving
"""

import pandas as pd
import numpy as np
from collections import namedtuple
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nnetsauce import PredictionInterval
from .utils import df_to_triangle, triangle_to_df


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
    conformal_method: a string
        conformal prediction method "splitconformal" or "localconformal"
    type_pi: a string;
        type of prediction interval: currently `None`
        split conformal prediction without simulation, "kde" or "bootstrap"
    use_factors : bool, default=False
        Whether to treat origin and development years as categorical variables
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, 
                 model=None, 
                 level=95,
                 replications=None,
                 conformal_method="splitconformal",
                 type_pi=None,
                 use_factors=False,
                 random_state=42):
        if model is None:
            model = RidgeCV(alphas=[10**i for i in range(-5, 5)])
        assert conformal_method in ("splitconformal", "localconformal"),\
              "must have conformal_method in ('splitconformal', 'localconformal')"    
        self.conformal_method = conformal_method
        self.model = PredictionInterval(model, level=level, 
                                        type_pi=type_pi, 
                                        type_split="sequential",
                                        method=conformal_method,
                                        replications=replications)
        self.level = level 
        self.replications = replications
        self.type_pi = type_pi 
        self.use_factors = use_factors
        self.origin_col = None
        self.development_col = None
        self.value_col = None
        self.max_dev = None
        self.origin_years = None
        self.cumulated = None 
        self.latest_ = None
        self.ultimate_ = None
        self.ultimate_lower_ = None
        self.ultimate_upper_ = None
        self.ibnr_mean_ = None
        self.ibnr_lower_ = None
        self.ibnr_upper_ = None
        self.X_test_ = None 
        self.full_data_ = None 
        self.full_data_upper_ = None 
        self.full_data_lower_ = None 
        self.full_data_sims_ = []
        self.scaler = StandardScaler()
        self.origin_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.dev_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def fit(self, data, origin_col="origin", 
            development_col="development", 
            value_col="values", 
            cumulated=True):
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
        cumulated: bool, default=True
            If the triangle is cumulated
            
        Returns
        -------
        self : object
            Returns self
        """
        # Store column names
        self.origin_col = origin_col
        self.development_col = development_col
        self.value_col = value_col
        self.cumulated = cumulated
        
        df = data.copy()
        df["dev"] = df[development_col] - df[origin_col] + 1
        df["calendar"] = df[origin_col] + df["dev"] - 1
        df = df.sort_values([origin_col, "dev"])

        # If data is cumulated, convert to incremental first
        if self.cumulated:
            # Calculate incremental values
            df[value_col] = df.groupby(origin_col)[value_col].diff().fillna(method='bfill')

        self.max_dev = df["dev"].max()
        self.origin_years = df[origin_col].unique()
        
        # Create full grid of all possible combinations
        full_grid = pd.MultiIndex.from_product(
            [self.origin_years, range(1, self.max_dev + 1)],
            names=[origin_col, "dev"]
        ).to_frame(index=False)
        
        # Merge with original data
        full_data = pd.merge(
            full_grid, 
            df[[origin_col, "dev", value_col]], 
            on=[origin_col, "dev"], 
            how="left"
        )
        
        # Calculate calendar year
        full_data["calendar"] = full_data[origin_col] + full_data["dev"] - 1
        
        # Calculate latest values for each origin year
        self.latest_ = full_data.groupby(origin_col)[value_col].last()
        
        # Apply transformations
        if self.use_factors:
            # One-hot encode origin and development years
            origin_encoded = self.origin_encoder.fit_transform(full_data[[origin_col]])
            dev_encoded = self.dev_encoder.fit_transform(full_data[["dev"]])
            
            # Create feature names for the encoded columns
            origin_feature_names = [f"origin_{year}" for year in self.origin_years]
            dev_feature_names = [f"dev_{i}" for i in range(1, self.max_dev + 1)]
            
            # Add encoded features to the dataframe
            full_data = pd.concat([
                full_data,
                pd.DataFrame(origin_encoded, columns=origin_feature_names, index=full_data.index),
                pd.DataFrame(dev_encoded, columns=dev_feature_names, index=full_data.index)
            ], axis=1)
            
            # Add calendar year as a feature
            full_data["log_calendar"] = np.log(full_data["calendar"])
            feature_cols = origin_feature_names + dev_feature_names + ["log_calendar"]
        else:
            # Use log transformations
            full_data["log_origin"] = np.log(full_data[origin_col])
            full_data["log_dev"] = np.log(full_data["dev"])
            full_data["log_calendar"] = np.log(full_data["calendar"])
            feature_cols = ["log_origin", "log_dev", "log_calendar"]
        
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
        
        Returns
        -------
        DescribeResult
            Named tuple containing mean, lower, and upper triangles
        """
        preds = self.model.predict(self.X_test_, return_pi=True)            
        
        to_predict = self.full_data_["to_predict"]        

        # Transform predictions back to original scale
        mean_pred = inv_arcsinh(preds.mean)
        lower_pred = inv_arcsinh(preds.lower)
        upper_pred = inv_arcsinh(preds.upper)
        
        # Store predictions in the full data
        self.full_data_.loc[to_predict, self.value_col] = mean_pred
        self.full_data_lower_.loc[to_predict, self.value_col] = lower_pred
        self.full_data_upper_.loc[to_predict, self.value_col] = upper_pred

        # Calculate IBNR based on predicted values (in incremental form)
        test_data = self.full_data_[to_predict]
        
        # Group by origin year and sum predictions
        self.ibnr_mean_ = test_data.groupby(self.origin_col)[self.value_col].sum()
        self.ibnr_lower_ = self.full_data_lower_[to_predict].groupby(self.origin_col)[self.value_col].sum()
        self.ibnr_upper_ = self.full_data_upper_[to_predict].groupby(self.origin_col)[self.value_col].sum()

        # If data was originally cumulated, convert predictions back to cumulative
        if self.cumulated:
            for df in [self.full_data_, self.full_data_lower_, self.full_data_upper_]:
                # Calculate cumulative values
                df[self.value_col] = df.groupby(self.origin_col)[self.value_col].cumsum()

        # Calculate triangles using utility function
        mean_triangle = df_to_triangle(
            self.full_data_,
            origin_col=self.origin_col,
            development_col="dev",
            value_col=self.value_col
        )
        lower_triangle = df_to_triangle(
            self.full_data_lower_,
            origin_col=self.origin_col,
            development_col="dev",
            value_col=self.value_col
        )
        upper_triangle = df_to_triangle(
            self.full_data_upper_,
            origin_col=self.origin_col,
            development_col="dev",
            value_col=self.value_col
        )

        # Calculate ultimate values
        if self.cumulated:
            # For cumulative data, ultimate is the last value in each origin year
            self.ultimate_ = self.full_data_.groupby(self.origin_col)[self.value_col].last()
            self.ultimate_lower_ = self.full_data_lower_.groupby(self.origin_col)[self.value_col].last()
            self.ultimate_upper_ = self.full_data_upper_.groupby(self.origin_col)[self.value_col].last()
        else:
            # For incremental data, ultimate is latest + IBNR
            self.ultimate_ = self.latest_ + self.ibnr_mean_
            self.ultimate_lower_ = self.latest_ + self.ibnr_lower_
            self.ultimate_upper_ = self.latest_ + self.ibnr_upper_

        DescribeResult = namedtuple("DescribeResult", 
                                    ("mean", "lower", "upper"))
        return DescribeResult(mean_triangle.T, 
                                lower_triangle.T, 
                                upper_triangle.T)
            
    def get_ibnr(self):
        """
        Get the IBNR (Incurred But Not Reported) values for each origin year
        
        Returns
        -------
        pandas.DataFrame
            IBNR values (mean, lower, upper) indexed by origin year
        """
        if self.ibnr_mean_ is None:
            raise ValueError("Model must be fitted and predict() must be called before getting IBNR values")

        DescribeResult = namedtuple("DescribeResult", 
                                    ("mean", "lower", "upper")) 
           
        return DescribeResult(self.ibnr_mean_, self.ibnr_lower_, self.ibnr_upper_)
        
    def get_latest(self):
        """
        Get the latest known values for each origin year
        
        Returns
        -------
        pandas.Series
            Latest known values indexed by origin year
        """
        if self.latest_ is None:
            raise ValueError("Model must be fitted before getting latest values")
        return self.latest_
        
    def get_ultimate(self):
        """
        Get the ultimate loss estimates for each origin year
        
        Returns
        -------
        pandas.DataFrame
            Ultimate loss estimates (mean, lower, upper) indexed by origin year
        """
        if self.ultimate_ is None:
            raise ValueError("Model must be fitted before getting ultimate values")
    
        DescribeResult = namedtuple("DescribeResult", 
                                      ("mean", "lower", "upper"))   
                     
        return DescribeResult(self.ultimate_,
                              self.ultimate_lower_,
                              self.ultimate_upper_)

    def get_summary(self):
        """
        Get a summary of reserving results including latest values, ultimate estimates,
        and IBNR values with confidence intervals.
        
        Returns
        -------
        dict
            Dictionary containing two keys:
            - 'ByOrigin': DataFrame with results by origin year
            - 'Totals': Series with total values
        """
        if self.ultimate_ is None:
            raise ValueError("Model must be fitted before getting summary")

        # Get latest values
        latest = self.get_latest()
        
        # Get ultimate values
        ultimate = self.get_ultimate()
        
        # Get IBNR values
        ibnr = self.get_ibnr()
        
        # Create summary by origin
        summary_by_origin = pd.DataFrame({
            'Latest': latest,
            'Mean Ultimate': ultimate.mean,
            'Mean IBNR': ibnr.mean,
            f'IBNR {self.level}%': ibnr.upper,
            f'Ultimate Lo{self.level}': ultimate.lower,
            f'Ultimate Hi{self.level}': ultimate.upper
        })
        
        # Calculate totals
        totals = pd.Series({
            'Latest': latest.sum(),
            'Mean Ultimate': ultimate.mean.sum(),
            'Mean IBNR': ibnr.mean.sum(),
            f'Total IBNR {self.level}%': ibnr.upper.sum()
        })
        
        return {
            'ByOrigin': summary_by_origin,
            'Totals': totals
        }