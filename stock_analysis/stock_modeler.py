"""Simple time series modeling for stocks."""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

class StockModeler:
    """Static methods for modeling stocks."""

    def __init__(self):
        raise NotImplementedError(
            "This class is to be used statically, don't instanstiate it!"
            )

    @staticmethod
    def decompose(df, freq, model='additive'):
        """
        Decompose the closing price of the stock into trend, seasonal,
        and remainder components.

        Parameters:
            - df: The dataframe containing the stock closing price as `close`
                  and with a time index.
            - freq: The number of periods in the frequency.
            - model: How to compute the decomposition
                     ('additive', 'multiplicative')

        Returns:
            A statsmodels decomposition object.
        """
        return seasonal_decompose(df.close, model=model, freq=freq)

    @staticmethod
    def arima(df, ar, i, ma, fit=True):
        """
        Create an ARIMA object for modeling time series.

        Parameters:
            - df: The dataframe containing the stock closing price as `close`
                  and with a time index.
            - ar: The autoregressive order (p).
            - i: The differenced order (q).
            - ma: The moving average order (d).
            - fit: Whether or not to return the fitted model,
                   defaults to True.

        Returns:
            A statsmodels ARIMA object which you can use to fit and predict.
        """
        arima_model = ARIMA(
            df.close.asfreq('B').fillna(method='ffill'), order=(ar, i, ma)
        )
        return arima_model.fit() if fit else arima_model

    @staticmethod
    def arima_residuals(arima_model_fitted):
        """
        Visualize the residuals from the model.

        Parameters:
            - arima_model_fitted: The fitted ARIMA model

        Returns:
            A matplotlib figure object.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        residuals = pd.Series(
            arima_model_fitted.resid, name='residuals'
        )
        residuals.plot(style='bo', ax=axes[0], title='Residuals')
        residuals.plot(kind='kde', ax=axes[1], title='Residuals KDE')
        plt.close()
        return fig

    @staticmethod
    def arima_predictions(arima_model_fitted, start, end, df, plot=True, **kwargs):
        """
        Get ARIMA predictions as pandas Series or plot.

        Parameters:
            - arima_model_fitted: The fitted ARIMA model.
            - start: The start date for the predictions.
            - end: The end date for the predictions.
            - df: The dataframe for the stock.
            - plot: Whether or not to plot the result, default is
                    True meaning the plot is returned instead of the
                    pandas Series containing the predictions.
            - kwargs: Additional keyword arguments to pass to the pandas
                      `plot()` method.

        Returns:
            A matplotlib Axes object or predictions as a Series
            depending on the value of the `plot` argument.
        """
        predicted_changes = arima_model_fitted.predict(
            start=start,
            end=end
        )

        predictions = pd.Series(
            predicted_changes, name='close'
        ).cumsum() + df.last('1D').close.iat[0]

        if plot:
            ax = df.close.plot(**kwargs)
            predictions.plot(ax=ax, style='r:', label='arima predictions')
            ax.legend()

        return ax if plot else predictions

    @staticmethod
    def regression(df):
        """
        Create linear regression of time series data with a lag of 1.

        Parameters:
            - df: The dataframe with the stock data.

        Returns:
            X, Y, and the fitted statsmodels linear regression
        """
        X = df.close.shift().dropna()
        Y = df.close[1:]
        return X, Y, sm.OLS(Y, X).fit()

    @staticmethod
    def regression_predictions(model, start, end, df, plot=True, **kwargs):
        """
        Get linear regression predictions as pandas Series or plot.

        Parameters:
            - model: The fitted linear regression model.
            - start: The start date for the predictions.
            - end: The end date for the predictions.
            - df: The dataframe for the stock.
            - plot: Whether or not to plot the result, default is
                    True meaning the plot is returned instead of the
                    pandas Series containing the predictions.
            - kwargs: Additional keyword arguments to pass to the pandas
                      `plot()` method.

        Returns:
            A matplotlib Axes object or predictions as a Series
            depending on the value of the `plot` argument.
        """
        predictions = pd.Series(
            index=pd.date_range(start, end),
            name='close'
        )
        last = df.last('1D').close
        for i, date in enumerate(predictions.index):
            if i == 0:
                pred = model.predict(last)
            else:
                pred = model.predict(predictions.iloc[i-1])
            predictions.loc[date] = pred[0]

        if plot:
            ax = df.close.plot(**kwargs)
            predictions.plot(
                ax=ax, style='r:', label='regression predictions'
            )
            ax.legend()

        return ax if plot else predictions
