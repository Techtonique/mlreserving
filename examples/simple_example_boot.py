"""
Simple example using the RAA dataset with MLReserving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mlreserving import MLReserving
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV

# Load the dataset
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/genins.csv"
df = pd.read_csv(url)

print(df.head())
print(df.tail())
df["values"] = df["values"]/1000

models = [RidgeCV(), ExtraTreesRegressor(), RandomForestRegressor()]

for mdl in models: 
    # Initialize the model with prediction intervals
    model = MLReserving(model=mdl,
        level=95,  # 80% confidence level
        type_pi="bootstrap",
        replications=250,
        random_state=42
    )        

    # Fit the model
    model.fit(df, origin_col="origin", 
              development_col="development",
                value_col="values")

    # Make predictions with intervals
    result = model.predict()
    ibnr = model.get_ibnr()

    print("\nMean predictions:")
    print(result.mean)
    print("\nIBNR per origin year (mean):")
    print(ibnr.mean)

    print("\nLower bound (95%):")
    print(result.lower)
    print("\nIBNR per origin year (lower):")
    print(ibnr.lower)

    print("\nUpper bound (95%):")
    print(result.upper)
    print("\nIBNR per origin year (upper):")
    print(ibnr.upper)

    # Display results
    print("\nMean predictions:")
    result.mean.plot()
    plt.title(f'Mean Predictions - {mdl.__class__.__name__}')
    plt.show()

    print("\nLower bound (95%):")
    result.lower.plot()
    plt.title(f'Lower Bound - {mdl.__class__.__name__}')
    plt.show()

    print("\nUpper bound (95%):")
    result.upper.plot()
    plt.title(f'Upper Bound - {mdl.__class__.__name__}')
    plt.show()

    # Plot IBNR
    plt.figure(figsize=(10, 6))
    plt.plot(ibnr.mean.index, ibnr.mean.values, 'b-', label='Mean IBNR')
    plt.fill_between(ibnr.mean.index, 
                        ibnr.lower.values, 
                        ibnr.upper.values, 
                        alpha=0.2, 
                        label='95% Confidence Interval')
    plt.title(f'IBNR per Origin Year - {mdl.__class__.__name__}')
    plt.xlabel('Origin Year')
    plt.ylabel('IBNR')
    plt.legend()
    plt.grid(True)
    plt.show()