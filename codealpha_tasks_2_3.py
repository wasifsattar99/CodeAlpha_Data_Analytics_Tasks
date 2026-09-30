"""
CodeAlpha Data Analytics Internship
Task 2: Exploratory Data Analysis (EDA)
Task 3: Data Visualization

Dataset: scikit-learn Diabetes dataset
The CSV is included in this project, so no internet connection is required.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes_dataset.csv")

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:")
print(df.describe())

# EDA: outlier check using IQR
numeric_cols = df.select_dtypes(include=np.number).columns
q1 = df[numeric_cols].quantile(0.25)
q3 = df[numeric_cols].quantile(0.75)
iqr = q3 - q1
outliers = ((df[numeric_cols] < (q1 - 1.5 * iqr)) |
            (df[numeric_cols] > (q3 + 1.5 * iqr))).sum()
print("\nPotential outliers by IQR:")
print(outliers.sort_values(ascending=False))

# Correlation analysis
corr = df.corr(numeric_only=True)["DiseaseProgressionScore"].drop("DiseaseProgressionScore")
print("\nCorrelation with target:")
print(corr.sort_values(ascending=False))

# Visualization 1
plt.figure(figsize=(9, 5))
plt.hist(df["DiseaseProgressionScore"], bins=25, edgecolor="black")
plt.title("Distribution of Disease Progression Score")
plt.xlabel("Disease Progression Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Visualization 2
corr_all = df.corr(numeric_only=True)
plt.figure(figsize=(11, 8))
plt.imshow(corr_all, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_all.columns)), corr_all.columns, rotation=70, ha="right")
plt.yticks(range(len(corr_all.index)), corr_all.index)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Visualization 3
top_feature = corr.abs().sort_values(ascending=False).index[0]
plt.figure(figsize=(9, 5))
plt.scatter(df[top_feature], df["DiseaseProgressionScore"], alpha=0.65)
plt.title(f"{top_feature} vs Disease Progression Score")
plt.xlabel(top_feature)
plt.ylabel("Disease Progression Score")
plt.tight_layout()
plt.show()

# Visualization 4
selected = ["bmi", "bp", "s5", "DiseaseProgressionScore"]
plt.figure(figsize=(10, 6))
plt.boxplot([df[c] for c in selected], labels=selected)
plt.title("Boxplots of Selected Variables")
plt.ylabel("Value")
plt.tight_layout()
plt.show()

# Visualization 5
df["BMI_Group"] = pd.qcut(df["bmi"], q=4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])
group_means = df.groupby("BMI_Group", observed=False)["DiseaseProgressionScore"].mean()
plt.figure(figsize=(9, 5))
group_means.plot(kind="bar")
plt.title("Average Disease Progression Score by BMI Quartile")
plt.xlabel("BMI Quartile")
plt.ylabel("Average Disease Progression Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
