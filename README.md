

# Stock Prophet 
This repo includes the files and instructions to deploy a stock prediciton model as a RESTful API using [FastAPI](https://fastapi.tiangolo.com/) to AWS EC2, and make it available (i.e., public) to end users. In the end we will containerize the FastAPI application that we built using Docker. Less focus is put on how well the model performs, yet the goal is to get an initial working system quickly into production:
    
    data -> model -> API -> deployment

Instructions can be found [here](Instructions.md).


## Objectives

- Develop a RESTful API with FastAPI
- Build a basic time series model to predict stock prices
- Deploy a FastAPI to AWS EC2
- Containerize the FastAPI application using Docker



## Getting prediction on AWS EC2    
After deploying on AWS, find the Public IPv4 address for your instance, e.g., 52.32.56.182, you can run the following in a shell on your local machine. You can use a different `ticker` and `days`:

```
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"ticker":"MSFT", "days":7}' \
http://52.32.56.182:8000/predict
```

## Getting prediction on the Docker image
### Build

The included Dockerfile allows for building a minimal container environment to run the API.

To build a docker image from the project, run the following command from the root project directory:

`docker build -t stock-prophet .`

### Run
To create a new container, run the following command:

`docker run -d --name mycontainer -p 8000:8000 stock-prophet`

### Test
To test whether the API service is running and functional.

`curl localhost:8000/ping`

### Get prediction


    curl \
    --header "Content-Type: application/json" \
    --request POST \
    --data '{"ticker":"MSFT", "days":7}' \
    http://localhost:8000/predict

