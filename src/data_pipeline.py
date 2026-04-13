import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import warnings

warnings.filterwarnings('ignore')

def fetch_market_data(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.Series]:
    """Fetches ETF pricing and risk-free rates."""
    tickers = ['SPY', 'TLT', 'SHY']
    prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    fred_data = web.DataReader('TB3MS', 'fred', start_date, end_date)
    fred_data.index = pd.to_datetime(fred_data.index).tz_localize(None)
    rf_daily = (fred_data['TB3MS'] / 100 / 252).reindex(prices.index).ffill()
    
    return prices, rf_daily

def load_and_decay_fill_pmi(filepath: str, halflife: int = 10, max_staleness: int = 45) -> pd.Series:
    """
    Loads manual PMI data and applies exponential decay fill for missing daily values.
    (Drop your specific PMI logic here).
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    # Apply your halflife logic here
    return df['PMI']