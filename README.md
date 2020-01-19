# machine-learning-prediction
Machine learning algorithm in Python that predicts a stock’s future price pattern based on historical data

Instructions:

1. Go to the working directory where you want to store the Python and .csv files.
2. Download machine-learning-prediction.py and AAPL.csv in the desired working directory.
3. Run Python script using the command: python machine-learning-prediction.py

Results:

Scatter plot of stock price predicted values vs. stock price actual values:
![PredictActualScatter](PredictActualScatter.png)

Text output: feature importance percentage array & fitting scores (decision tree vs. gradient-boosting)
![MLOutput](MLOutput.png)

Comments:
- For the scatter plot of predicted prices and actual prices, data closely fitting the line y=x indicates the predictions are accurate to the actual values.
- Too high of a performance score means the model is overfitting the training data, which results in a poor prediciton of new values (test data).