# ARIMA
In this project we aim at estimating Bitcoin price based on its history using ARIMA model.
Towards this end, our repository contains 4 files, including part1, part2, part3, and part 4.
Part1 includes loading packages, reading data, checking its stationarity, and performing preprocessing for making it linear. As the reader will see, we use log and diffrential transformations for making the history data stationary.

Part2 includes performing a grid search on (p,d,q) values, fitting the data with any tuple of them, finding feasible tuples (i.e. the ones whose  resdiduals have no special trend) and saving the achieved RSS from each fitting. At the end, the best (p,d,q) tuple, w.r.t the RSS value, is presented.

Part3 includes performing a grid search on feasible (p,d,q) values, loading their fiting data (from part2 results), selecting K random start locations for estimation window, applying the ARIMA(p,d,q) model to the window, estimating the value of bitcoin price for the next day, and saving the MSE results.


