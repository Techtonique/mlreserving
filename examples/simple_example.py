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

    # Fit the model
    model.fit(df, origin_col="origin", development_col="development", value_col="values")

    # Make predictions with intervals
    result = model.predict()

    print("result.mean", result.mean)
    print("result.lower", result.lower)
    print("result.upper", result.upper)

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