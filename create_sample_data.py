import pandas as pd
import numpy as np

# Create sample creditcard.csv for demo
np.random.seed(42)

# Generate 5000 sample transactions with more fraud cases
n_samples = 5000
data = {
    'Time': np.random.randint(0, 86400, n_samples),
    'Amount': np.random.exponential(100, n_samples),
    'Class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud rate
}

# Add V1-V28 features (PCA components)
for i in range(1, 29):
    data[f'V{i}'] = np.random.normal(0, 1, n_samples)

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv('creditcard.csv', index=False)
print("✅ Sample creditcard.csv created!")
print(f"📊 {len(df)} transactions generated")
print(f"🚨 Fraud rate: {df['Class'].mean():.1%}")