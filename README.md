# MLReserving

A machine learning-based reserving model for (longitudinal data) insurance claims.

## Installation

```bash
pip install mlreserving
```

## Usage

```python
from mlreserving import MLReserving
import pandas as pd

# Create your triangle data
# Load the dataset
url = "https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/triangle/raa.csv"
data = pd.read_csv(url)

# Initialize and fit the model
model = MLReserving(model=mdl,
                    level=80,  # 80% confidence level
                    random_state=42)
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
