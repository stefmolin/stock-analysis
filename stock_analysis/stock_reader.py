"""Gathering select stock data."""

import datetime
import re

import pandas as pd
import pandas_datareader.data as web

from .utils import label_sanitizer

class StockReader:
    """Class for reading financial data from websites."""

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
            lambda x: x.strftime('%Y%m%d') if isinstance(x, datetime.date) \
            else re.sub(r'\D', '', x),
            [start, end or datetime.date.today()]
        )

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
        return web.DataReader(ticker, 'iex', self.start, self.end)

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
            parse_dates=True,
            index_col='Date'
        )[0]

    @label_sanitizer
    def get_sp500_data(self):
        """
        Get historical OHLC data from Yahoo Finance for the S&P 500 Index
        for given date range.

        Returns:
            A pandas dataframe with the S&P 500 index data.
        """
        return web.get_data_yahoo('^GSPC', self.start, self.end)
