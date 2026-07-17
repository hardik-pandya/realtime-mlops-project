#Generate Synethetic Churn Data Set

import pandas as pd
import numpy as np

# Setting the random seed to ensure reproducibility of the synthetic churn dataset
np.random.seed(42)

#Generate 1000 samples
n_samples = 1000

data = {
    'customer_id': np.arange(1, n_samples + 1),
    'age': np.random.randint(18, 70, size=n_samples),
    'tenure_months': np.random.randint(1, 72, size=n_samples),
    'monthly_charges': np.random.uniform(20.0, 120.0, size=n_samples),
    'total_charges': np.round(np.random.uniform(100.0, 8000, size=n_samples)),
    'num_support_calls': np.random.randint(0, 10, size=n_samples),
}

#Simple Churn Logic: higer monthly charges and more support calls increase churn probability
churn_probability = (
    (data['monthly_charges'] / 120.0) * 0.3 +
    (data['num_support_calls'] / 10.0) * 0.4 +
    (1 - (data['tenure_months'] / 72.0)) * 0.3
)
data['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)

df = pd.DataFrame(data)
df.to_csv('data/churn_data.csv', index=False)
print(f"Generated synthetic churn dataset with {n_samples} samples and saved to 'data/churn_data.csv'.")
print(f"Churn rate: {df['churn'].mean():.2%}")