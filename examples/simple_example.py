"""
Simple example using the RAA dataset with MLReserving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mlreserving import MLReserving
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV

# Load the dataset
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/genins.csv"
df = pd.read_csv(url)

print(df.head())
print(df.tail())

models = [RidgeCV(), ExtraTreesRegressor()]

for mdl in models: 
    # Initialize the model with prediction intervals
    model = MLReserving(model=mdl,
        level=95,  # 95% confidence level
        random_state=42
    )

    df["values"] = df["values"]/1000

    # Fit the model
    model.fit(df, origin_col="origin", development_col="development", value_col="values")

    # Make predictions with intervals
    result = model.predict()

    print("\nMean predictions:")
    print(result.mean)
    print("\nIBNR per origin year (mean):")
    print(result.ibnr_mean)

    print("\nLower bound (95%):")
    print(result.lower)
    print("\nIBNR per origin year (lower):")
    print(result.ibnr_lower)

    print("\nUpper bound (95%):")
    print(result.upper)
    print("\nIBNR per origin year (upper):")
    print(result.ibnr_upper)

    # Display results
    print("\nMean predictions:")
    result.mean.plot()
    plt.show()

    print("\nLower bound (95%):")
    result.lower.plot()
    plt.show()

    print("\nUpper bound (95%):")
    result.upper.plot()
    plt.show()

    # Plot IBNR
    plt.figure(figsize=(10, 6))
    plt.plot(result.ibnr_mean.index, result.ibnr_mean.values, 'b-', label='Mean IBNR')
    plt.fill_between(result.ibnr_mean.index, 
                    result.ibnr_lower.values, 
                    result.ibnr_upper.values, 
                    alpha=0.2, 
                    label='95% Confidence Interval')
    plt.title('IBNR per Origin Year')
    plt.xlabel('Origin Year')
    plt.ylabel('IBNR')
    plt.legend()
    plt.grid(True)
    plt.show()