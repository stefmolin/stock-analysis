"""Gathering select stock data."""

import datetime
import re

import pandas as pd
import pandas_datareader.data as web

from .utils import label_sanitizer

class StockReader:
    """Class for reading financial data from websites."""

    _index_tickers = {
        'SP500' : '^GSPC',
        'DOW' : '^DJI',
        'NASDAQ' : '^IXIC'
    }

    def __init__(self, start, end=None):
        """
        Create a StockReader object for reading across a given date range.

        Parameters:
            - start: The first date to include, as a datetime object or
                     a string in the format 'YYYYMMDD'.
            - end: The last date to include, as a datetime object or
                   a string in the format 'YYYYMMDD'. Defaults to today
                   if not provided.

        Returns:
            A StockReader object.
        """
        self.start, self.end = map(
            lambda x: x.strftime('%Y%m%d') if isinstance(
                x, datetime.date
            ) else re.sub(r'\D', '', x),
            [start, end or datetime.date.today()]
        )

    @property
    def available_tickers(cls):
        """Access the names of the indices whose tickers are supported."""
        return cls._index_tickers.keys()

    @classmethod
    def get_index_ticker(cls, index):
        """
        Class method for getting the ticker of the specified index, if known.

        Parameters:
            - index: The name of the index; check `cls.available_tickers`
                     for full list which includes:
                         - 'SP500' for S&P 500,
                         - 'DOW' for Dow Jones Industrial Average,
                         - 'NASDAQ' for NASDAQ Composite Index

        Returns:
            The ticker as a string if known, otherwise None.
        """
        try:
            index = index.upper()
        except:
            raise ValueError('`index` must be a string')
        return cls._index_tickers.get(index, None)

    @label_sanitizer
    def get_ticker_data(self, ticker):
        """
        Get historical OHLC data from Investors Exchange (IEX)
        for given date range and ticker.

        Parameter:
            - ticker: The stock symbol to lookup as a string.

        Returns:
            A pandas dataframe with the stock data.
        """
        data = web.DataReader(ticker, 'iex', self.start, self.end)
        data.index = pd.to_datetime(data.index)
        return data

    @label_sanitizer
    def get_bitcoin_data(self):
        """
        Get bitcoin historical OHLC data from coinmarketcap.com for given date range.

        Returns:
            A pandas dataframe with the bitcoin data.
        """
        return pd.read_html(
            'https://coinmarketcap.com/currencies/bitcoin/historical-data/?'
            'start={}&end={}'.format(
                self.start, self.end
            ),
            parse_dates=[0],
            index_col=[0]
        )[0].sort_index()

    @label_sanitizer
    def get_index_data(self, index='SP500'):
        """
        Get historical OHLC data from Yahoo Finance for the chosen index
        for given date range.

        Parameter:
            - index: String representing the index you want data for,
                     supported indices include:
                        - 'SP500' for S&P 500,
                        - 'DOW' for Dow Jones Industrial Average,
                        - 'NASDAQ' for NASDAQ Composite Index
                    Check the `available_tickers` property for more.

        Returns:
            A pandas dataframe with the index data.
        """
        if index not in self.available_tickers:
            raise ValueError(
                'Index not supported. '
                f"Available tickers are: {', '.join(self.available_tickers)}"
            )
        return web.get_data_yahoo(
            self.get_index_ticker(index), self.start, self.end
        )
