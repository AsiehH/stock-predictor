from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from model import predict, convert

app = FastAPI()

@app.get("/ping")
def pong():
    return {"ping": "pong!"}
# pydantic models
class StockIn(BaseModel):
    ticker: str
    days: int

class StockOut(StockIn):
    forecast: dict

@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    ticker = payload.ticker
    days = payload.days

    prediction_list = predict(ticker, days)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {
        "ticker": ticker, 
        "days": days,
        "forecast": convert(prediction_list)}
    return response_object