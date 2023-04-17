import datetime
from pathlib import Path
from typing import Optional, List, Dict


import joblib
import pandas as pd
import yfinance as yf
from prophet import Prophet

import argparse

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("models")
FIG_DIR = Path(BASE_DIR).joinpath("figures")
TODAY = datetime.date.today()


def train(ticker: str="MSFT") -> None:
    """
    Downloads historical data from Yahoo Finance, trains a Prophet model, and saves it using joblib.

    Args:
        ticker (str): Ticker symbol of the stock to be used for training the model. Defaults to "MSFT".

    Returns:
        None
    """
    # Download historical data
    data = yf.download(ticker, "2020-01-01", TODAY.strftime("%Y-%m-%d"))
    # Prepare data for modeling
    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"]
    df_forecast["y"] = df_forecast["Adj Close"]
    df_forecast = df_forecast[["ds", "y"]]
    # Fit Prophet model
    model = Prophet()
    model.fit(df_forecast)
    # Save trained model
    joblib.dump(model, Path(MODEL_DIR).joinpath(f"{ticker}.joblib"))


def predict(ticker: str="MSFT", days: int=7) -> Optional[List[Dict[str, float]]]:
    """
    Predict the stock price of a given ticker using the trained Prophet model.

    Parameters
    ----------
    ticker : str, optional
        Ticker symbol of the stock to predict. Default is "MSFT".
    days : int, optional
        Number of days to forecast. Default is 7.

    Returns
    -------
    Optional[List[Dict[str, float]]]
        If the model file exists, returns a list of dictionaries with the predicted dates and prices for the next `days`.
        Otherwise, returns `None`.

    """
    model_file = Path(MODEL_DIR).joinpath(f"{ticker}.joblib")
    if not model_file.exists():
        return False

    model = joblib.load(model_file)

    future = TODAY + datetime.timedelta(days=days)

    dates = pd.date_range(start="2020-01-01", end=future.strftime("%m/%d/%Y"),)
    df = pd.DataFrame({"ds": dates})

    forecast = model.predict(df)

    model.plot(forecast).savefig(
        Path(FIG_DIR).joinpath(f"{ticker}_plot.png"))
    model.plot_components(forecast).savefig(
        Path(FIG_DIR).joinpath(f"{ticker}_plot_components.png"))

    return forecast.tail(days).to_dict("records")

def convert(prediction_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Convert the list of dictionaries returned by the `predict` function to a dictionary with dates as keys and predicted prices as values.

    Parameters
    ----------
    prediction_list : List[Dict[str, float]]
        A list of dictionaries with predicted dates and prices.

    Returns
    -------
    Dict[str, float]
        A dictionary with dates as keys and predicted prices as values.

    """    
    output = {}
    for data in prediction_list:
        date = data["ds"].strftime("%m/%d/%Y")
        output[date] = data["trend"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--ticker', type=str, default='MSFT', help='Stock Ticker')
    parser.add_argument('--days', type=int, default=7, help='Number of days to predict')
    args = parser.parse_args()
    
    train(args.ticker)
    prediction_list = predict(ticker=args.ticker, days=args.days)
    output = convert(prediction_list)
    print(output)