"""Simple time series modeling for stocks."""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

from .utils import validate_df


class StockModeler:
    """Static methods for modeling stocks."""

    def __init__(self):
        raise NotImplementedError(
            "This class must used statically; don't instantiate it."
        )

    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def decompose(df, period, model='additive'):
        """
        Decompose the closing price of the stock into trend, seasonal,
        and remainder components.

        Parameters:
            - df: The dataframe containing the stock closing price as `close`
                  and with a time index.
            - period: The number of periods in the frequency.
            - model: How to compute the decomposition
                     ('additive', 'multiplicative')

        Returns:
            A `statsmodels` decomposition object.
        """
        return seasonal_decompose(df.close, model=model, period=period)

    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def arima(df, *, ar, i, ma, fit=True, freq='B'):
        """
        Create an ARIMA object for modeling time series.

        Parameters:
            - df: The dataframe containing the stock closing price as `close`
                  and with a time index.
            - ar: The autoregressive order (p).
            - i: The differenced order (q).
            - ma: The moving average order (d).
            - fit: Whether or not to return the fitted model,
                   defaults to `True`.
            - freq: The frequency of the data. Default is 1 business day ('B').

        Returns:
            A `statsmodels` ARIMA object which you can use to fit and predict.
        """
        arima_model = ARIMA(
            df.close.asfreq(freq).fillna(method='ffill'), order=(ar, i, ma)
        )
        return arima_model.fit() if fit else arima_model

    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def arima_predictions(df, arima_model_fitted, start, end, plot=True, **kwargs):
        """
        Get ARIMA predictions as a `pandas.Series` object or plot.

        Parameters:
            - df: The dataframe for the stock.
            - arima_model_fitted: The fitted ARIMA model.
            - start: The start date for the predictions.
            - end: The end date for the predictions.
            - plot: Whether or not to plot the result, default is
                    `True` meaning the plot is returned instead of the
                    `pandas.Series` object containing the predictions.
            - kwargs: Additional keyword arguments to pass to the pandas
                      `plot()` method.

        Returns:
            A matplotlib `Axes` object or predictions as a `pandas.Series`
            object depending on the value of the `plot` argument.
        """
        predictions = arima_model_fitted.predict(start=start, end=end)

        if plot:
            ax = df.close.plot(**kwargs)
            predictions.plot(ax=ax, style='r:', label='arima predictions')
            ax.legend()

        return ax if plot else predictions

    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def regression(df):
        """
        Create linear regression of time series data with a lag of 1.

        Parameters:
            - df: The dataframe with the stock data.

        Returns:
            X, Y, and the fitted `statsmodels` linear regression
        """
        X = df.close.shift().dropna()
        Y = df.close[1:]
        return X, Y, sm.OLS(Y, X).fit()

    @staticmethod
    @validate_df(columns={'close'}, instance_method=False)
    def regression_predictions(df, model, start, end, plot=True, **kwargs):
        """
        Get linear regression predictions as a `pandas.Series` object or plot.

        Parameters:
            - df: The dataframe for the stock.
            - model: The fitted linear regression model.
            - start: The start date for the predictions.
            - end: The end date for the predictions.
            - plot: Whether or not to plot the result, default is
                    `True` meaning the plot is returned instead of the
                    `pandas.Series` object containing the predictions.
            - kwargs: Additional keyword arguments to pass to the pandas
                      `plot()` method.

        Returns:
            A matplotlib `Axes` object or predictions as a `pandas.Series`
            object depending on the value of the `plot` argument.
        """
        predictions = pd.Series(
            index=pd.date_range(start, end),
            name='close'
        )
        last = df.last('1D').close
        for i, date in enumerate(predictions.index):
            if not i:
                pred = model.predict(last)
            else:
                pred = model.predict(predictions.iloc[i - 1])
            predictions.loc[date] = pred[0]

        if plot:
            ax = df.close.plot(**kwargs)
            predictions.plot(
                ax=ax, style='r:', label='regression predictions'
            )
            ax.legend()

        return ax if plot else predictions

    @staticmethod
    def plot_residuals(model_fitted, freq='B'):
        """
        Visualize the residuals from the model.

        Parameters:
            - model_fitted: The fitted model
            - freq: Frequency that the predictions were made on.
                    Default is 'B' (business day).

        Returns:
            A matplotlib `Axes` object.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        residuals = pd.Series(
            model_fitted.resid.asfreq(freq), name='residuals'
        )
        residuals.plot(style='bo', ax=axes[0], title='Residuals')
        axes[0].set(xlabel='Date', ylabel='Residual')
        residuals.plot(kind='kde', ax=axes[1], title='Residuals KDE')
        axes[1].set_xlabel('Residual')
        return axes
