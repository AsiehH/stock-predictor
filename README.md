# stock-predictor

The instruction for creating this API is found on my [FourthBrain repo](https://github.com/AsiehH/MLE-7/tree/main/assignments/week-11-intro-mlops/stock-predictor)

To get predictions from the API, run the following command from a terminal window on your local machine. The IP@ is the public IPV4 of your EC2 instance. 

    ```
    curl \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{"ticker":"MSFT", "days":7}' \
    http://52.32.56.182:8000/predict
    ```
    
Alternatively, you can access the API documentaiton by using your public IPV4 on this url: 
**http://52.32.56.182:8000/docs** 

Result example:    

<p align="center">
<img src="img/docs01.png" alt="drawing" width="400"/>

<img src="img/docs02.png" alt="drawing" width="240"/>
</p>

    
    
# Questions

## Algorithm Understanding
- How does the Prophet Algorithm differ from an LSTM?
Why does an LSTM have poor performance against ARIMA and Profit for Time Series?

	- The LSTM prediction is based on a set of last values, we are therefore less prone to variance due to seasonality and already consider the current trend.	
In contrast to that, the prophet model is an additive system and finds out and displays seasonalities.

## Interview Readiness
- What is exponential smoothing and why is it used in Time Series Forecasting?

	- Exponential smoothing is a time series forecasting method for univariate data. Forecasts produced using exponential smoothing methods are weighted averages of past observations, with the weights decaying exponentially as the observations get older. In other words, the more recent the observation the higher the associated weight.

- What is stationarity? What is seasonality? Why Is Stationarity Important in Time Series Forecasting?
	- 	stationarity in a time series data means that the time series does not have trend or seasonality, has constant variance over time and constant autocorrelation structure. In other words, the observations in a stationary time series are not dependent on time. Statistical modeling methods assume or require the time series to be stationary to be effective.

	- Seasonality is a characteristic of a time series in which the data experiences regular and predictable changes that recur every period, which can be a year, n-years or even less, depending on the data.

	
- How is seasonality different from cyclicality? Fill in the blanks:
_Seasonality_ is predictable, whereas _cyclicality_ is not.	