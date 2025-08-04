# Portfolio-Optimization
Designed an optimized investment portfolio using historical price data of selected stocks. • Applied mean-variance analysis to calculate efficient frontier and maximize Sharpe Ratio. • Used Python (Pandas, NumPy, Matplotlib) for data processing • Identified optimal asset allocations that minimized risk for a given level of expected return.
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize


#Step1 : Define tickers and time range. Define the lists of tickers
tickers = ["SPY", "BND", "GLD", "QQQ", "VTI"]
#Set the end date to today
end_date = datetime.today()
#Set the start date to 5 years ago
start_date = end_date- timedelta(days = 5*365)
print(start_date)

#Step 3 Download adjusted close prices 
#Create an empty DataFrame to store the adjusted close prices
adj_close_df = pd.DataFrame()

#Download the close prices for each ticker 
for ticker in tickers:
  data = yf.download(ticker, start = start_date, end = end_date)
  adj_close_df[ticker] = data['Adj Close']

#Display the DataFrame
print(adj_close_df)

#Step 3 :Calculate lognormal returns 
#Calculate the lognormal returns for each ticker 
log_returns = np.log(adj_close_df/adj_close_df.shift(1)) #one days price divided by the previous days price

#Drop any missing values
log_returns = log_returns.dropna()

#Sec 4: Calculate Covariance Matrix
#Calculate the covariance matrix using annualized log returns (Learn covariance and how much of it is good)
#Covariance matrix is very imp because this is how we measure the total risk of portfolio
cov_matrix = log_returns.cov()*252
print(cov_matrix) #this tells us covariance between each of the asset i.e SPY has covar of 0.004970 with GLD

#Define Portfolio performance metrics 
#Calculate the portfolio standard deviation, The line of code calculates the portfolio variance, which is a measure of the risk associated with a portfolio of assets. It represents the combined volatility of the assets
# in the portfolio taking into account their individual volatilities and correlations with each other.(@ symbol means you r multiplying two rows together)
def standard_deviation(weights, cov_matrix):
  variance = weights.T @ cov_matrix @ weights
  return np.sqrt(variance)

#Calculate the expected return
#Key-Assumption: expected returns are based on historical returns 
def expected_return(weights, log_returns):
  return np.sum(log_returns.mean()*weights)*252

#Calculate the Sharpe Ration, Sharpe ratio is the expected return minus risk free rate, so basically the risk premium 
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
  return(expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
  

risk_free_rate = 0.02

!pip install fredapi
#Portfolio Optimization, Set the risk- free rate
#Step 1- finding risk free rate using FRED API key website for best results

from fredapi import Fred # import fredapi module
#Replace 'your_api_key' with your actual FRED API key
fred = Fred(api_key = '0923ec6232356e75b1813e305d00ab7d')
ten_year_treasury_rate = fred.get_series_latest_release('DGS10')/100
#set the risk free rate
risk_free_rate = ten_year_treasury_rate.iloc[-1]
print(risk_free_rate) #showing the current 10 year treasury rate as 4.3%

#Define the function to minimize (negative Sharpe Ratio)
#In the case of the scipy.optimize.minimize(function, there is no direct method to find the max value of a function)
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
  return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

#Set the constraits and bounds: Constrains are conditions that must met by the soln during the optimization process.
#In essence, these constraints and bounds ensure that:
#Weights Sum to 1: The total allocation across all assets is 100%.
#Weight Limits: No individual asset weight can exceed 50%.
constraints = {'type': 'eq', 'fun': lambda weights:np.sum(weights)-1}
bounds = [(0,0.5) for _ in range(len(tickers))] #setting bound 0 means we cannot go short in any of these assets, we cannot sell any asset that we don't own.
#upper bound 0.5 means we can't have our portfolio more than 50% in any single securities within it.

#set the initial weights
initial_weights = np.array([1/len(tickers)]*len(tickers))
print(initial_weights) #output is 20% for each securities

#Optimize the weights to maximize Sharpe Ratio
#'SLSQP' stands for sequential least squares quadratic programming, which is a numerical optimization technique suitable for solving
#nonlinear optimization problems with constraints
optimized_results = minimize(neg_sharpe_ratio, initial_weights, args = (log_returns, cov_matrix, risk_free_rate),
                                 method = 'SLSQP', bounds = bounds, constraints = constraints)

#Get the optimal weights
optimal_weights = optimized_results.x

#Analyze the optimal portfolio
#Display analytics of the optimal portfolio


print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
  print(f"{ticker}: {weight:.4f}")

print()
optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print(f"Expected annual Return: {optimal_portfolio_return:.4f}")
print(f"Annual Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")


#Display the final portfolio in a plot
import matplotlib.pyplot as plt
#Import the required library
import matplotlib.pyplot as plt 
#Create a bar chart of the optimal weights
plt.figure(figsize = (10,6))
plt.bar(tickers, optimal_weights)
#add labels and a title
plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')
plt.show()
