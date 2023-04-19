

# Stock Prophet 
This repo includes the files and instructions to deploy a stock prediciton model as a RESTful API using [FastAPI](https://fastapi.tiangolo.com/) to AWS EC2, and make it available (i.e., public) to end users. Less focus is put on how well the model performs, yet the goal is to get an initial working system quickly into production:
    
    data -> model -> API -> deployment


## Objectives

- Develop a RESTful API with FastAPI
- Build a basic time series model to predict stock prices
- Deploy a FastAPI to AWS EC2



## Result    
In the end, after finding the Public IPv4 address for your instance, e.g., 52.32.56.182, you can run the following in a shell on your local machine. You can use a different `ticker` and `days`:

```
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"ticker":"MSFT", "days":7}' \
http://52.32.56.182:8000/predict
```



