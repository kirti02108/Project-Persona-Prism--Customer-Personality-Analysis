# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. Data Cleaning and Feature Engineering ---

# Load the dataset
try:
    # THIS IS THE CORRECTED LINE THAT FIXES THE ERROR
    data = pd.read_csv('marketing_campaign.csv', sep='\t')
except FileNotFoundError:
    print("Error: 'marketing_campaign.csv' not found. Please download the dataset.")
    # As a fallback, create a dummy dataframe to allow the rest of the code to run for demonstration.
    data = pd.DataFrame({
        'ID': range(100), 'Year_Birth': np.random.randint(1950, 2000, 100),
        'Education': ['Graduation']*100, 'Marital_Status': ['Single']*100,
        'Income': np.random.randint(20000, 80000, 100), 'Kidhome': [0]*100,
        'Teenhome': [0]*100, 'Dt_Customer': ['2014-01-01']*100,
        'MntWines': np.random.randint(0, 1000, 100), 'MntFruits': [0]*100,
        'MntMeatProducts': [0]*100, 'MntFishProducts': [0]*100,
        'MntSweetProducts': [0]*100, 'MntGoldProds': [0]*100,
    })

# Calculate Age and Total Spending
data['Age'] = 2024 - data['Year_Birth']
data['Total_Spending'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

# Calculate Seniority (how long they've been a customer)
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
# Note: Using 2025 as a reference date for calculation
data['Seniority'] = (pd.to_datetime('2025-01-01') - data['Dt_Customer']).dt.days

# Simplify Marital Status and Education
data['Marital_Status'] = data['Marital_Status'].replace(['Married', 'Together'], 'In Relationship')
data['Marital_Status'] = data['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], 'Single')
data['Education'] = data['Education'].replace(['PhD', 'Master', 'Graduation'], 'Postgraduate')
data['Education'] = data['Education'].replace(['2n Cycle', 'Basic'], 'Undergraduate')

# Create a feature for number of children
data['Children'] = data['Kidhome'] + data['Teenhome']

# Drop unnecessary columns and rows with missing income
# Note: Added 'Z_CostContact' and 'Z_Revenue' from your dataset columns to the drop list
data = data.drop(['ID', 'Year_Birth', 'Dt_Customer', 'Kidhome', 'Teenhome',
                  'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                  'MntSweetProducts', 'MntGoldProds', 'Z_CostContact', 'Z_Revenue'], axis=1)
data = data.dropna(subset=['Income'])


# --- 2. Customer Segmentation with K-Means Clustering ---

# Select features for clustering
features = data[['Income', 'Total_Spending', 'Seniority', 'Age', 'Children']]

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # Set n_init explicitly
clusters = kmeans.fit_predict(scaled_features)
data['Cluster'] = clusters

# --- 3. Visualization and Interpretation ---

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Income', y='Total_Spending', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segments by Income and Spending')
plt.xlabel('Income')
plt.ylabel('Total Spending')
plt.legend(title='Customer Segment')
plt.grid(True)
plt.show()


# Analyze the characteristics of each cluster
cluster_summary = data.groupby('Cluster')[['Income', 'Total_Spending', 'Seniority', 'Age', 'Children']].mean().round(0)
print("\n--- Cluster Summary (Averages) ---")
print(cluster_summary)
print("\n[Done] Script finished successfully.")