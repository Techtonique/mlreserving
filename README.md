# MLReserving

A machine learning based reserving model for insurance claims.

## Installation

```bash
pip install mlreserving
```

## Usage

```python
from mlreserving import MLReserving
import pandas as pd

# Create your triangle data
data = pd.DataFrame({
    'origin': [...],  # Origin years
    'development': [...],  # Development years
    'values': [...]  # Claim values
})

# Initialize and fit the model
model = MLReserving()
model.fit(data)

# Make predictions
result = model.predict()

# Get IBNR, latest, and ultimate values
ibnr = model.get_ibnr()
latest = model.get_latest()
ultimate = model.get_ultimate()
```

## Features

- Machine learning based reserving model
- Support for prediction intervals
- Flexible model selection
- Handles both continuous and categorical features

## License

BSD Clause Clear License
