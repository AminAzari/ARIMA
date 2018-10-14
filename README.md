# ARIMA
In this project, we aim at estimating Bitcoin price based on its history using ARIMA model.
Towards this end, our repository contains 3 main files, including part1, part2, and part3, as well as several data files
.
Part1 includes loading packages, reading data, checking its stationarity, and performing preprocessing for making it linear. As the reader will see, we use log and differential transformations for making the history data stationary.

Part2 includes performing a grid search on (p,d,q) values, fitting the data with any tuple of them, finding feasible tuples (i.e. the ones whose  residuals have no special trend) and saving the achieved RSS from each fitting. In the end, the best (p,d,q) tuple, w.r.t the RSS value, is presented.

Part3 includes performing a grid search on feasible (p,d,q) values, loading their fitting data (from part2 results), selecting K random start locations for estimation window, applying the ARIMA(p,d,q) model to the window, estimating the value of bitcoin price for the next day, and saving the MSE results.

The datan.csv contains the bitcoin price table (closing price in USD) for a priod od 3 years, from 01-09-2015 to 31-08-2018.

The auxiliary data files contain the results of fitting different (p,d,q) tuples to the data.




