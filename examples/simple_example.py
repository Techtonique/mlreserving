"""
Simple example using the RAA dataset with MLReserving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mlreserving import MLReserving

# Load the dataset
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/abc.csv"
df = pd.read_csv(url)

print(df.head())
print(df.tail())

# Initialize the model with prediction intervals
model = MLReserving(
    level=95,  # 95% confidence level
    #replications=250,  # number of replications for simulation
    #type_pi="bootstrap",  # use bootstrap for prediction intervals
    random_state=42
)

# Fit the model
model.fit(df, origin_col="origin", development_col="development", value_col="values")

# Make predictions with intervals
result = model.predict()

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