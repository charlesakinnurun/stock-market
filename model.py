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


