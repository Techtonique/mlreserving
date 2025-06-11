# MLReserving

A machine learning-based reserving model for (longitudinal data) insurance claims.

![PyPI](https://img.shields.io/pypi/v/mlreserving) [![PyPI - License](https://img.shields.io/pypi/l/mlreserving)](./LICENSE) [![Downloads](https://pepy.tech/badge/mlreserving)](https://pepy.tech/project/mlreserving) 
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/mlreserving/)


## Installation

```bash
pip install mlreserving
```

## Usage

```python
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
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/raa.csv"
df = pd.read_csv(url)

print(df.head())
print(df.tail())


models = [RidgeCV(), ExtraTreesRegressor(), RandomForestRegressor()]

# Try both factor and non-factor approaches
for use_factors in [False, True]:
    print(f"\n{'='*50}")
    print(f"Using {'factors' if use_factors else 'log transformations'}")
    print(f"{'='*50}\n")
    
    for mdl in models: 
        # Initialize the model with prediction intervals
        model = MLReserving(model=mdl,
            level=80,  # 80% confidence level
            use_factors=use_factors,  # Use categorical encoding
            random_state=42
        )        

        # Fit the model
        model.fit(df, origin_col="origin", development_col="development", value_col="values")

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

        print("\n Summary:")
        print(model.get_summary())

        # Display results
        print("\nMean predictions:")
        result.mean.plot()
        plt.title(f'Mean Predictions - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.show()

        print("\nLower bound (95%):")
        result.lower.plot()
        plt.title(f'Lower Bound - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.show()

        print("\nUpper bound (95%):")
        result.upper.plot()
        plt.title(f'Upper Bound - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.show()

        # Plot IBNR
        plt.figure(figsize=(10, 6))
        plt.plot(ibnr.mean.index, ibnr.mean.values, 'b-', label='Mean IBNR')
        plt.fill_between(ibnr.mean.index, 
                         ibnr.lower.values, 
                         ibnr.upper.values, 
                         alpha=0.2, 
                         label='95% Confidence Interval')
        plt.title(f'IBNR per Origin Year - {mdl.__class__.__name__} ({use_factors and "Factors" or "Log"})')
        plt.xlabel('Origin Year')
        plt.ylabel('IBNR')
        plt.legend()
        plt.grid(True)
        plt.show()
```

## Features

- Machine learning based reserving model
- Support for prediction intervals
- Flexible model selection
- Handles both continuous and categorical features

## License

BSD Clause Clear License
