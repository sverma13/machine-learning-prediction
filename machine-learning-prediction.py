# Data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import statsmodels.api as sm

# Machine learning libraries
import keras
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score

# Load stock data
ticker = 'AAPL'
tickerFile = ticker + '.csv'
stockData = pd.read_csv(tickerFile, parse_dates=['Date'])
stockData = stockData.sort_values(by='Date')
stockData.set_index('Date', inplace=True)

# Define returns
#returns = stockData['Adj Close'].pct_change().dropna()
#stockData['Returns'] = returns

stockData['10 Day Close Pct'] = stockData['Adj Close'].pct_change(10)

# Define technical indicators
rsi14 = ta.momentum.rsi(close=stockData['Adj Close'], n=14, fillna=False)
stockData['RSI_14'] = rsi14

# Define machine learning features & targets
shift = -10

stockData = stockData.dropna()

features = stockData[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '10 Day Close Pct', 'RSI_14']]
features = features.iloc[:shift]
targets = stockData['10 Day Close Pct'].shift(shift)
targets = targets.iloc[:shift]

trainSize = int(0.85 * targets.shape[0])
trainFeatures = features[:trainSize]
trainTargets = targets[:trainSize]
testFeatures = features[trainSize:]
testTargets = targets[trainSize:]





# Neural network setup
decisionTree = DecisionTreeRegressor(max_depth=5)
decisionTree.fit(trainFeatures, trainTargets)

trainPredictions = decisionTree.predict(trainFeatures)
testPredictions = decisionTree.predict(testFeatures)

treeTrain = decisionTree.score(trainFeatures, trainTargets)
treeTest = decisionTree.score(testFeatures, testTargets)
print('Neural network train score:', treeTrain)
print('Neural network test score:', treeTest)

plt.figure()
plt.scatter(trainPredictions, trainTargets, label='Train')
plt.scatter(testPredictions, testTargets, label='Test')
plt.title('Neural Network - Train vs Test Data')
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.legend()
plt.show()





# Random forest
randomForest = RandomForestRegressor(n_estimators=200, max_depth=5)
randomForest.fit(trainFeatures, trainTargets)
trainPredictions = randomForest.predict(trainFeatures)
testPredictions = randomForest.predict(testFeatures)

forestTrain = randomForest.score(trainFeatures, trainTargets)
forestTest = randomForest.score(testFeatures, testTargets)
print('Random forest train score:', forestTrain)
print('Random forest test score:', forestTest)
#print('Feature importance array:', randomForest.feature_importances_)

plt.figure()
plt.scatter(trainPredictions, trainTargets, label='Train')
plt.scatter(testPredictions, testTargets, label='Test')
plt.title('Random Forest - Train vs Test Data')
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.legend()
plt.show()






# Gradient boosting
gbr = GradientBoostingRegressor(learning_rate=0.01, n_estimators=200, subsample=0.6)
gbr.fit(trainFeatures, trainTargets)
trainPredictions = gbr.predict(trainFeatures)
testPredictions = gbr.predict(testFeatures)

gbrTrain = gbr.score(trainFeatures, trainTargets)
gbrTest = gbr.score(testFeatures, testTargets)
print('Gradient boosting train score:', gbrTrain)
print('Gradient boosting test score:', gbrTest)

plt.figure()
plt.scatter(trainPredictions, trainTargets, label='Train')
plt.scatter(testPredictions, testTargets, label='Test')
plt.title('Gradient Boosting - Train vs Test Data')
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.legend()
plt.show()


#Scale data
sc = StandardScaler()
scaledTrainFeatures = sc.fit_transform(trainFeatures)
scaledTestFeatures = sc.transform(testFeatures)

gbr = GradientBoostingRegressor(learning_rate=0.01, n_estimators=200, subsample=0.6)
gbr.fit(scaledTrainFeatures, trainTargets)
trainPredictions = gbr.predict(scaledTrainFeatures)
testPredictions = gbr.predict(scaledTestFeatures)

gbrTrain = gbr.score(scaledTrainFeatures, trainTargets)
gbrTest = gbr.score(scaledTestFeatures, testTargets)
print('SCALED Gradient boosting train score:', gbrTrain)
print('SCALED Gradient boosting test score:', gbrTest)

plt.figure()
plt.scatter(trainPredictions, trainTargets, label='Train')
plt.scatter(testPredictions, testTargets, label='Test')
plt.title('SCALED Gradient Boosting - Train vs Test Data')
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.legend()
plt.show()






"""
plt.subplot(2,1,1)
plt.hist(trainFeatures, bins=20)
plt.subplot(2,1,2)
plt.hist(scaledTrainFeatures, bins=5)
plt.show()
"""
# Keras on scaled data (originally 50 relu, 10 relu, 1 linear)
model = Sequential()
model.add(Dense(50, input_dim=scaledTrainFeatures.shape[1], activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
history = model.fit(scaledTrainFeatures, trainTargets, epochs=50)
model.fit(scaledTrainFeatures, trainTargets)

plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()


trainPredictions = model.predict(scaledTrainFeatures)
testPredictions = model.predict(scaledTestFeatures)

kerasTrain = r2_score(trainTargets, trainPredictions)
kerasTest = r2_score(testTargets, testPredictions)
print('SCALED Keras train score:', kerasTrain)
print('SCALED Keras test score:', kerasTest)

plt.figure()
plt.scatter(trainPredictions, trainTargets, label='Train')
plt.scatter(testPredictions, testTargets, label='Test')
plt.title('SCALED Keras - Train vs Test Data')
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.legend()
plt.show()
