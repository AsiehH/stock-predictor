
# Stock Prophet Deployment 

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



# Build Container Image

## Interview Readiness

- What does it mean to create a Docker image and why do we use Docker images?

	- Docker is a software that allows us to put our applicaiton inside of containers. We create a docker image to include our applicaiton and its dependencies so as to avoid facing the hiccups resulting from environment changes. 
	
- Please explain what is the difference between a Container vs a Virtual Machine?
	- VM virtualize an entire machine down to the hardware layers and containers only virtualize software layers above the operating system level.
	- containers are more portable and efficient
	- containers take up less space than VMs
	- containers can handle more applications 

- What are 5 examples of container orchestration tools (please list tools)?
 	- Kubernetes
 	- Apache Mesos
 	- Docker Swarm
 	- Openshift
 	- Rancher
 	- Managed Container Orchestration tools
	 	- 	Google container engine (GKE)
	 	-  Google Cloud Run
	 	-  AWS Elastic Kubernetes Service (EKS)
	 	-  Amazon EC2 Conainer Service (ECS)
	 	-  Azue AKS Service

- How does a Docker image differ from a Docker container?
	- A `Dockerfile` is a text file that Docker reads in from top to bottom. It contains a bunch of instructions which informs Docker HOW the Docker image should get built.
	- 	A docker image gets built by running a Docker command (which uses a docker file). 
	- A Docker container is a running instance of a Docker image.
