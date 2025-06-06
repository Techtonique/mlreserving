"""
Utility functions for mlreserving package
"""

import pandas as pd
import numpy as np


def df_to_triangle(df, origin_col="origin", development_col="development", value_col="values"):
    """
    Convert a data frame with origin, development, and value columns into a triangle format
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data with origin, development, and value columns
    origin_col : str, default="origin"
        Name of the origin year column
    development_col : str, default="development"
        Name of the development year column
    value_col : str, default="values"
        Name of the value column
        
    Returns
    -------
    pandas.DataFrame
        Triangle format with origin years as index and development years as columns
    """
    # Calculate development lag and calendar year
    df = df.copy()
    
    # If development_col is not 'dev', calculate it
    if development_col != "dev":
        df["dev"] = df[development_col] - df[origin_col] + 1
    
    df["calendar"] = df[origin_col] + df["dev"] - 1
    
    # Create triangle
    triangle = df.pivot(
        index=origin_col,
        columns="dev",
        values=value_col
    ).sort_index()
    
    return triangle

def triangle_to_df(triangle, origin_col="origin", 
                   development_col="development", 
                   value_col="values"):
    """
    Convert a triangle format into a data frame with origin, development, and value columns
    
    Parameters
    ----------
    triangle : pandas.DataFrame
        Triangle format with origin years as index and development years as columns
    origin_col : str, default="origin"
        Name of the origin year column
    development_col : str, default="development"
        Name of the development year column
    value_col : str, default="values"
        Name of the value column
        
    Returns
    -------
    pandas.DataFrame
        Data frame with origin, development, and value columns
    """
    # Reset index to get origin years as a column
    df = triangle.reset_index()
    
    # Melt the development columns into rows
    df = pd.melt(
        df,
        id_vars=[origin_col],
        var_name="dev",
        value_name=value_col
    )
    
    # Calculate development year and calendar year
    df[development_col] = df[origin_col] + df["dev"] - 1
    df["calendar"] = df[origin_col] + df["dev"] - 1
    
    # Reorder columns and sort by calendar year
    df = df[[origin_col, development_col, "dev", "calendar", value_col]]

    df.sort_values("calendar", inplace=True)
    
    return df
