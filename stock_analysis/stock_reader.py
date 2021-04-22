"""Gather select stock data."""

import datetime as dt
import re

import pandas as pd
import pandas_datareader.data as web

from .utils import label_sanitizer


class StockReader:
    """Class for reading financial data from websites."""

    _index_tickers = {
        'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'NASDAQ': '^IXIC', # US
        'S&P/TSX Composite Index': '^GSPTSE', # Canada
        'IPC Mexico': '^MXX', # Mexico
        'IBOVESPA': '^BVSP', # Brazil
        'Euro Stoxx 50': '^STOXX50E', # Europe
        'FTSE 100': '^FTSE', # UK
        'CAC 40': '^FCHI', # France
        'DAX': '^GDAXI', # Germany
        'IBEX 35': '^IBEX', # Spain
        'FTSE MIB': 'FTSEMIB.MI', # Italy
        'OMX Stockholm 30': '^OMX', # Sweden
        'Swiss Market Index': '^SSMI', # Switzerland
        'Nikkei': '^N225', # Japan
        'Hang Seng': '^HSI', # Hong Kong
        'CSI 300': '000300.SS', # China
        'S&P BSE SENSEX': '^BSESN', # India
        'S&P/ASX 200': '^AXJO', # Australia
        'MOEX Russia Index': '^IMOEX.ME' # Russia
    } # to add more, consult https://finance.yahoo.com/world-indices/

    def __init__(self, start, end=None):
        """
        Create a `StockReader` object for reading across a given date range.

        Parameters:
            - start: The first date to include, as a datetime object or
                     a string in the format 'YYYYMMDD'.
            - end: The last date to include, as a datetime object or
                   a string in the format 'YYYYMMDD'. Defaults to today
                   if not provided.

        Returns:
            A `StockReader` object.
        """
        self.start, self.end = map(
            lambda x: \
                x.strftime('%Y%m%d') if isinstance(x, dt.date)\
                else re.sub(r'\D', '', x),
            [start, end or dt.date.today()]
        )
        if self.start >= self.end:
            raise ValueError('`start` must be before `end`')

    @property
    def available_tickers(self):
        """Access the names of the indices whose tickers are supported."""
        return list(self._index_tickers.keys())

    @classmethod
    def get_index_ticker(cls, index):
        """
        Class method for getting the ticker of the specified index, if known.

        Parameters:
            - index: The name of the index; check `available_tickers`
                     property for full list which includes:
                         - 'S&P 500' for S&P 500,
                         - 'Dow Jones' for Dow Jones Industrial Average,
                         - 'NASDAQ' for NASDAQ Composite Index

        Returns:
            The ticker as a string if known, otherwise `None`.
        """
        try:
            index = index.upper()
        except AttributeError:
            raise ValueError('`index` must be a string')
        return cls._index_tickers.get(index, None)

    @label_sanitizer
    def get_ticker_data(self, ticker):
        """
        Get historical OHLC data for given date range and ticker
        from Yahoo! Finance.

        Parameter:
            - ticker: The stock symbol to lookup as a string.

        Returns:
            A `pandas.DataFrame` object with the stock data.
        """
        return web.get_data_yahoo(ticker, self.start, self.end)


    def get_index_data(self, index):
        """
        Get historical OHLC data from Yahoo! Finance for the chosen index
        for given date range.

        Parameter:
            - index: String representing the index you want data for,
                     supported indices include:
                        - 'S&P 500' for S&P 500,
                        - 'Dow Jones' for Dow Jones Industrial Average,
                        - 'NASDAQ' for NASDAQ Composite Index
                    Check the `available_tickers` property for more.

        Returns:
            A `pandas.DataFrame` object with the index data.
        """
        if index not in self.available_tickers:
            raise ValueError(
                'Index not supported. '
                f"Available tickers are: {', '.join(self.available_tickers)}"
            )
        return self.get_ticker_data(self.get_index_ticker(index))


    def get_bitcoin_data(self, currency_code):
        """
        Get bitcoin historical OHLC data for given date range.

        Parameter:
            - currency_code: The currency to collect the bitcoin data 
                             in, e.g. USD or GBP.

        Returns:
            A `pandas.DataFrame` object with the bitcoin data.
        """
        return self.get_ticker_data(f'BTC-{currency_code}').loc[self.start:self.end]


    def get_risk_free_rate_of_return(self, last=True):
        """
        Get the risk-free rate of return using the 10-year US Treasury bill.
        Source: FRED (https://fred.stlouisfed.org/series/DGS10)

        Parameter:
            - last: If `True` (default), return the rate on the last date in the date range
                    else, return a `Series` object for the rate each day in the date range.

        Returns:
            A single value or a `pandas.Series` object with the risk-free rate(s) of return.
        """
        data = web.DataReader('DGS10', 'fred', start=self.start, end=self.end)
        data.index.rename('date', inplace=True)
        data = data.squeeze()
        return data.asof(self.end) if last and isinstance(data, pd.Series) else data


    @label_sanitizer
    def get_forex_rates(self, from_currency, to_currency, **kwargs):
        """
        Get daily foreign exchange rates from AlphaVantage.

        Note: This requires an API key, which can be obtained for free at
        https://www.alphavantage.co/support/#api-key. To use this method, you must either
        store it as an environment variable called `ALPHAVANTAGE_API_KEY` or pass it in to
        this method as `api_key`.

        Parameters:
            - from_currency: The currency you want the exchange rates for.
            - to_currency: The target currency.

        Returns:
            A `pandas.DataFrame` with daily exchange rates.
        """
        data = web.DataReader(
            f'{from_currency}/{to_currency}', 'av-forex-daily',
            start=self.start, end=self.end, **kwargs
        ).rename(pd.to_datetime)
        data.index.rename('date', inplace=True)
        return data
