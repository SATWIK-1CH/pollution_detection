# pollution_detection
Pollution forecasting

The Air pollution is a serious concern in All the countries. The problem should be tackled in an efficient manner as All the governments and citizens of the countries are showing high interest for the same. Air Quality index (AQI) is   a measure of pollution in air. Due to industrialization and increase in fossil fuels there is a tremendous 
increase in Air pollution in past few decades.The present study is mainly focusing on using Box-Jenkins ARIMA (Auto Regressive Integrated Moving Average model) for predicting a pollutant. stochastic ARIMA model has a strong potential for short term prediction. In this study we would be applying time series analysis on data from UCI website. 
. The order of best ARIMA model has been found out by carrying out different combinations of Akaike Information’s criterion, Bayesian Information criterion and prediction error along with auto correlation function and partial auto cross correlation function. ARIMA model assumes that Time series is Linear and the residual terms should follow a specific distribution known as NORMAL distribution. With help of ARIMA model the behavioural dynamics can be adjusted into single equation.
represents these lags.
There are 3 implementations that need to be performed while stating ARIMA mode
I.	Model identification: using plots of data of autocorrelation graphs, partial auto correlation graphs and other info, a set of parameter values are initialized for p, q .in our study we obtained p, q values as 5,2 respectively
II.	Model estimation:  In our study we have used CSS-MLE (conditional sum of square-maximum likelihood estimator) for estimating parameters checking values of AIC (AKAIKE INFORMATION CRITERION) AND BIC (BAYESIAN INFORMATION CRITERION) respectively, for an optimized model, the values of AIC and BIC are as low as possible.
III.	Diagnostic checking: The plots of ACF and PACF are observed and we can ensure p, q values as 7,3 respectively.

