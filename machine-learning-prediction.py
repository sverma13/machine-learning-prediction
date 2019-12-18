# Data analysis libraries
import pandas as pd
import numpy as np

# Machine learning libraries
import keras
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor

# Statistical computation library
import statsmodels.api as sm

ticker = 'AAPL'

# Load stock data
tickerFile = ticker + ".csv"
data = pd.read_csv(tickerFile, parse_dates=['Date'])
data = data.sort_values(by='Date')
data.set_index('Date', inplace=True)

data["Returns"] = data["Adj Close"].pct_change()
returns = data["Returns"].dropna()

# Define machine learning features & targets
features = data[['Returns', 'Volume']].dropna()
print(features)
targets = data['Adj Close'].shift(-10).pct_change().dropna()
print(targets)
print(type(features))
print(type(targets))

# Split up machine learning test and train sets
linearFeatures = sm.add_constant(features.values)
trainSize = int(0.85 * targets.shape[0])
trainFeatures = linearFeatures[:trainSize]
trainTargets = targets[:trainSize]
testFeatures = linearFeatures[trainSize:]
testTargets = targets[trainSize:]

# Create regression model
model = sm.OLS(trainTargets, trainFeatures)
results = model.fit()
print(results.summary())