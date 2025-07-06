import yfinance as yf
import pandas as pd

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "V", "JPM", "JNJ",
    "UNH", "HD", "PG", "MA", "AVGO", "PEP", "COST", "LLY", "MRK", "KO",
    "XOM", "CVX", "WMT", "DIS", "BAC", "ADBE", "CSCO", "TMO", "PFE", "INTC",
    "ABT", "CRM", "CMCSA", "MCD", "VZ", "ORCL", "ACN", "NKE", "QCOM", "DHR",
    "TXN", "LIN", "UPS", "MDT", "NEE", "IBM", "PM", "AMGN", "HON", "AMAT"
]

def get_stock_data(tickers, start='2018-01-01', end='2023-12-31'):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # This extracts the 'Close' prices for all tickers
    df = df.xs('Close', axis=1, level=0)

    return df.dropna(axis=1)

def load_spy_cumulative(start_date, end_date):
    import yfinance as yf
    spy = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close']
    spy_returns = spy.pct_change().fillna(0)
    spy_cumulative = (1 + spy_returns).cumprod() - 1
    return spy_cumulative


if __name__ == "__main__":
    df = get_stock_data(tickers)
    df.to_csv("data/price_data.csv")


    
