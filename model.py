# %% [markdown]
# Import the neccessary libraries

# %%
import yfinance as yf # Library to download historical stock data
import pandas as pd # Data manipulation and analysis library
import numpy as np # Numerical operations, especially for feature engineering
import matplotlib.pyplot as plt # Plotting library for visualizations
import seaborn as sns # Enhanced data visualization library
from sklearn.preprocessing import StandardScaler # Tool to standardize features
from sklearn.decomposition import PCA # Principal Component Analysis for dimensionality reduction (for 2D plotting)
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN # The three clustering algorithms to compare
from sklearn.metrics import silhouette_score # Evaluation metric for clustering

# %% [markdown]
# Data Acquistion

# %%
# Define a diverse list of stock tickers from various sectors
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","JPM","JNJ","KO","KO","XOM","TSLA","NFLX"]
# Define the date range for the historical data
START_DATE = "2020-01-01"
END_DATE = None

# %%
# Download the historical Adjusted Close prices for the defined tickers
print(f"Downloading data for {len(TICKERS)} stocks from {START_DATE} to {END_DATE}.........")
# The Adj Close price is used as it accounts for dividends and stock splits
stock_data = yf.download(TICKERS,start=START_DATE,end=END_DATE)

# %%
# Fill any potential missing values (NaN) that may occur due to market closures or data gaps
# ffill (forward fill) propagates the last valid observation forward
stock_data = stock_data.fillna(method="ffill")
print("----- Stock Data Head (Adjusted Close Prices) -----")
stock_data.head()

# %% [markdown]
# Data Preprocessing

# %%
df = stock_data

# %%
df

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("----- Missing Values -----")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("----- Duplicated Rows -----")
print(df_duplicated)

# %%
# Check the shape of the data
print("----- Shape of the Data -----")
print(df.shape)

# %% [markdown]
# Feature Engineering

# %%
# Calculate Daily Lag Returns
# Log returns are preferred over simple returns in finance for stationary and additive properties
log_returns = np.log(df / df.shift(1)).dropna()

# %%
# Calculate Annualized Mean Returns (Performance Features)
# Assumes approximately 252 trading days in a year
# Multiply the daily mean return by 252
mean_return = log_returns.mean() * 252

# %%
# Calculate Annualized Volatility (Risk Feature)
# Volatility is the Standard deviation of returns
# Multiply the daily standard devaition by the square root of 252
volatility = log_returns.std() * np.sqrt(252)

# %%
# Combine the two engineered features into a single DataFrame for clustering
features_df = pd.DataFrame({
    "Mean Return":mean_return,
    "Volatility":volatility
})

# %%
print("Engineered Features (Annualized Risk/Return):")
print(features_df)

# %%
# Extract the values (X) for scaling and modelling
X = features_df.values

# %% [markdown]
# Data Scaling

# %%
# Standardize the features
# Scaling is cruical for distance-based algorithms (K-Means,Agglomerative)
# It ensures that no single feature dominates the distance calculation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Apply PCA for 2D Visualization
# We reduce the 2 features down to 2 principal components (although redundant here,
# this step is standard practice for generalizing to higher dimension)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca,index=features_df.index,columns=["PC1","PC2"])

# %% [markdown]
# Pre-Training Visualization

# %%
plt.Figure(figsize=(10,6))
# Create a scatter plot of the stocks in the PC1 vs PC2 space
sns.scatterplot(x=pca_df["PC1"],y=pca_df["PC2"],s=100,color="grey",alpha=0.6)
# Add labels for each ticker on the plot
for i,ticker in enumerate(pca_df.index):
    plt.annotate(ticker,(pca_df["PC1"][i] + 0.05,pca_df["PC2"][i]),fontsize=10)
plt.title('Stock Clustering: Data Distribution Before Clustering (PCA)')
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.grid(True,linestyle="--",alpha=0.5)
plt.show()

# %% [markdown]
# Hyperparameter Tuning and Model Comparison

# %%
# Dictionary to store the results for comparison
results = {}
best_score = -1
best_model_name = ""
best_model = None

# %%
# ----- K-Means Clustering and Tunin (Elbow Method) -----
# Elbow Method to find the optimal K (number of clusters)
wcss = [] # WCSS (Within-Cluster Sum of Squares) list
k_range = range(2,min(10,len(TICKERS))) # Check K values from 2 up to the number of tickers
print("----- K-Means Elbow Method Tuning -----")

for k in k_range:
    # Initialize KMeans with k clusters, using "k-means++" for smart centroid initialization
    kmeans = KMeans(n_clusters=k,random_state=42,n_init="auto")
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # "inertia_ is the WCSS value"

# Plot WCSS to find the "elbow" (the point where the decrease rate slows down)
plt.Figure(figsize=(8,4))
plt.plot(k_range,wcss,marker="o",linestyle="--")
plt.title("Elbow Method for Optimal K (K-Means)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.grid(True,linestyle=":",alpha=0.5)
plt.show()

# %%
# Based on visualization (or heuristic, we pick a reasonable K=3)
optimal_k = 3
# Heuristic selection based on typical financial groupings (High-Risk/Return, Low-Risk/Return, Mid-Range)

# Final K-Means Model
kmeans_model = KMeans(n_clusters=optimal_k,random_state=42,n_init="auto").fit(X_scaled)
kmeans_score = silhouette_score(X_scaled,kmeans_model.labels_)
results["K-Means"] = {
    "Score":kmeans_score,
    "Labels":kmeans_model.labels_
}
print(f"K-Means (K={optimal_k}) Silhouette Score: {kmeans_score:>4f}")

# %%
# Update best model
if kmeans_score > best_score:
    best_score = kmeans_score
    best_model_name = "K-Means"
    best_model = kmeans_model

# %%
# ----- Hierarachical Clustering -----
# Hierarchical Clustering (Agglomerative) does not strictly need K,
# but for Silhouette Score Comparison, we use same optimal_k from K-Means
hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k,linkage="ward").fit(X_scaled)
# "ward" linkage minimizes the variane of the Clusters being merged
hierarchical_score = silhouette_score(X_scaled,hierarchical_model.labels_)
results["Hierarchical"] = {
    "Score":hierarchical_score,
    "Labels":hierarchical_model.labels_
}
print(f"Hierarchical (k={optimal_k}) Silhouette Score: {hierarchical_score:.4f}")

# %%
# Update best model
if hierarchical_score > best_score:
    best_score = hierarchical_score
    best_model_name = "Hierarchical"
    best_model = hierarchical_model

# %%
# ----- DSCAN Clustering and Tuning
# DBSCAN tuning involves finding optimal epilson (eps) and minimum samples (min_samples)
# We test a few common settings and select the best one based on Silhouette Score
# Note: DBSCAN often performs poorly on this type of data (global density) and may produce many noise points (-1 labels)
dbscan_parameters = [
    (0.5,2), # (eps,min_samples)
    (0.6,2),
    (0.4,3)
]
best_dbscan_score = -1

for eps,min_pts in dbscan_parameters:
    dbscan_model = DBSCAN(eps=eps,min_samples=min_pts).fit(X_scaled)
    # Exclude noise points (-1) from Silhouette calculation for a fairer score
    labels = dbscan_model.labels_
    if len(np.unique(labels)) > 1: # Must have more than 1 cluster to calculate score
        # Only use non-noise points for scoring (a practical adaptation for comparison)
        score = silhouette_score(X_scaled[labels != -1],labels[labels !=-1])
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_dbscan_model = dbscan_model
            best_dbscan_labels = labels

results["DBSCAN"] = {
    "Score":best_dbscan_score,
    "Labels":best_dbscan_labels
}
print(f"DBSCAN (Best Score): {best_dbscan_score:.4f}")

# %%
# Update best model
if best_dbscan_score > best_score:
    best_score = best_dbscan_score
    best_model_name = "DBSCAN"
    best_model = best_dbscan_model

# %% [markdown]
# Final Results, Visualization After Training

# %%
# Add the final cluster labels from the best model to the PCA DataFrame
pca_df["'Cluster"] = best_model.labels_
features_df["Cluster"] = best_model.labels_

print(f"\n=========================================================")
print(f"BEST PERFORMING MODEL: {best_model_name} with Silhouette Score: {best_score:.4f}")
print(f"Stock Cluster Assignments:")
print(features_df[["Cluster"]])
print(f"\n=========================================================")

# %%
# Visualization AFTER Clustering (using the best model's results)
plt.Figure(figsize=(10,6))
# Use the assigned cluster labels to color the scatter plot
sns.scatterplot(x=pca_df["PC1"],y=pca_df["PC2"],palette="viridis",s=150,legend="full")

# Add Annotations (stock tickers)
for i, ticker in enumerate(pca_df.index):
    plt.annotate(ticker, (pca_df['PC1'][i] + 0.05, pca_df['PC2'][i]), 
                 fontsize=10, weight='bold')
    
plt.title(f"Stock Clustering AFTER Trainiing: {best_model_name} Results")
plt.xlabel("Principal Components 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.legend(title="Cluster ID",loc="upper right")
plt.grid(True,linestyle="--",alpha=0.6)
plt.show()


