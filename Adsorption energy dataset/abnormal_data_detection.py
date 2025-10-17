import numpy as np
import pandas as pd
from scipy import stats

# Read adsorption energy data from CSV file
df = pd.read_csv('smi_pmf_PO4.csv')
adsorption_energy = df['adsorption_energy'].values

# Convert to numpy array for statistical analysis
adsorption_energy = np.array(adsorption_energy)

# Calculate basic statistical properties
print("Basic Statistics:")
print(f"Number of data points: {len(adsorption_energy)}")
print(f"Minimum value: {adsorption_energy.min():.2f}")
print(f"Maximum value: {adsorption_energy.max():.2f}")
print(f"Mean value: {adsorption_energy.mean():.2f}")
print(f"Median value: {np.median(adsorption_energy):.2f}")
print(f"Standard deviation: {adsorption_energy.std():.2f}")
print(f"Q1 (25th percentile): {np.percentile(adsorption_energy, 25):.2f}")
print(f"Q3 (75th percentile): {np.percentile(adsorption_energy, 75):.2f}")
print(f"IQR (Interquartile Range): {np.percentile(adsorption_energy, 75) - np.percentile(adsorption_energy, 25):.2f}")

# Detect outliers using IQR method
Q1 = np.percentile(adsorption_energy, 25)
Q3 = np.percentile(adsorption_energy, 75)
IQR = Q3 - Q1

# Calculate lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nOutlier detection bounds (IQR method):")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

# Identify low and high outliers
outliers_low = adsorption_energy[adsorption_energy < lower_bound]
outliers_high = adsorption_energy[adsorption_energy > upper_bound]

print(f"\nSignificant low outliers (< {lower_bound:.2f}): {len(outliers_low)}")
print(f"Significant high outliers (> {upper_bound:.2f}): {len(outliers_high)}")

# Identify specific outlier data points with their indices
outlier_indices = []
for i, value in enumerate(adsorption_energy):
    if value < lower_bound or value > upper_bound:
        outlier_indices.append((i, value))

print(f"\nSpecific outlier data points:")
for idx, value in outlier_indices:
    print(f"ID {idx}: {value:.2f}")

# Verify outliers using Z-score method
z_scores = np.abs(stats.zscore(adsorption_energy))
z_outliers = adsorption_energy[z_scores > 3]

print(f"\nOutliers detected by Z-score method (|Z| > 3): {len(z_outliers)}")
for value in z_outliers:
    idx = np.where(adsorption_energy == value)[0][0]
    z_score = z_scores[idx]
    print(f"ID {idx}: {value:.2f} (Z-score: {z_score:.2f})")