# Data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import statsmodels.api as sm

# Machine learning libraries
#import keras
#import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Load stock data
ticker = 'AAPL'
tickerFile = ticker + '.csv'
stockData = pd.read_csv(tickerFile, parse_dates=['Date'])
stockData = stockData.sort_values(by='Date')
stockData.set_index('Date', inplace=True)

# Define returns
returns = stockData['Adj Close'].pct_change().dropna()
stockData['Returns'] = returns

# Define technical indicators
rsi14 = ta.momentum.rsi(close=stockData['Adj Close'], n=14, fillna=False)
stockData['RSI_14'] = rsi14



# Define machine learning features & targets
shift = 10

stockData = stockData.dropna()

features = stockData[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Returns', 'RSI_14']]
features = features.iloc[:-shift]
targets = stockData['Adj Close'].shift(-shift)
targets = targets.iloc[:-shift]
"""
# Split up machine learning test and train sets
linearFeatures = sm.add_constant(features.values)
trainSize = int(0.85 * targets.shape[0])
trainFeatures = linearFeatures[:trainSize]
trainTargets = targets[:trainSize]
testFeatures = linearFeatures[trainSize:]
print(type(testFeatures))
testTargets = targets[trainSize:]
print(type(testTargets))

# Create regression model
model = sm.OLS(trainTargets, trainFeatures)
results = model.fit()
print(results.summary())
print(results.pvalues)
"""
trainSize = int(0.85 * targets.shape[0])
trainFeatures = features[:trainSize]
trainTargets = targets[:trainSize]
testFeatures = features[trainSize:]
testTargets = targets[trainSize:]

# Neural network setup
decisionTree = DecisionTreeRegressor(max_depth=5)
decisionTree.fit(trainFeatures, trainTargets)
#print(decisionTree.score(trainFeatures, trainTargets))
#print(decisionTree.score(testFeatures, testTargets))

trainPredictions = decisionTree.predict(trainFeatures)
testPredictions = decisionTree.predict(testFeatures)
plt.scatter(trainPredictions, trainTargets, label='train')
plt.scatter(testPredictions, testTargets, label='test')
plt.legend()
plt.show()

# Random forest
randomForest = RandomForestRegressor(n_estimators=200, max_depth=5)
randomForest.fit(trainFeatures, trainTargets)
print('Random forest score:', randomForest.score(trainFeatures, trainTargets))
print('Feature importance array:', randomForest.feature_importances_)

# Gradient boosting
gbr = GradientBoostingRegressor(learning_rate=0.01, n_estimators=200, subsample=0.06)
gbr.fit(trainFeatures, trainTargets)
print('Gradient boosting score:', gbr.score(trainFeatures, trainTargets))
