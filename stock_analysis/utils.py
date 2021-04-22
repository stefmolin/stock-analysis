"""Utility functions for stock analysis."""

from functools import wraps
import re

import pandas as pd


def _sanitize_label(label):
    """
    Clean up a label by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters:
        - label: The text you want to fix.

    Returns:
        The sanitized label.
    """
    return re.sub(r'[^\w\s]', '', label).lower().replace(' ', '_')


def label_sanitizer(method):
    """
    Decorator around a method that returns a dataframe to
    clean up all labels in said dataframe (column names and index name)
    by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters:
        - method: The method to wrap.

    Returns:
        A decorated method or function.
    """
    # keep the docstring of the data method for help()
    @wraps(method)
    def method_wrapper(self, *args, **kwargs):
        df = method(self, *args, **kwargs)

        # fix the column names
        df.columns = [
            _sanitize_label(col) for col in df.columns
        ]

        # fix the index name
        df.index.rename(
            _sanitize_label(df.index.name),
            inplace=True
        )
        return df
    return method_wrapper


def validate_df(columns, instance_method=True):
    """
    Decorator that raises a `ValueError` if input isn't a pandas
    `DataFrame` or doesn't contain the proper columns. Note the `DataFrame`
    must be the first positional argument passed to this method.

    Parameters:
        - columns: A set of required column names.
                   For example, {'open', 'high', 'low', 'close'}.
        - instance_method: Whether or not the item being decorated is
                           an instance method. Pass `False` to decorate
                           static methods and functions.

    Returns:
        A decorated method or function.
    """
    def method_wrapper(method):
        @wraps(method)
        def validate_wrapper(self, *args, **kwargs):
            # functions and static methods don't pass self
            # so self is the first positional argument in that case
            df = (self, *args)[0 if not instance_method else 1]
            if not isinstance(df, pd.DataFrame):
                raise ValueError('Must pass in a pandas `DataFrame`')
            if columns.difference(df.columns):
                raise ValueError(
                    f'DataFrame must contain the following columns: {columns}'
                )
            return method(self, *args, **kwargs)
        return validate_wrapper
    return method_wrapper


def group_stocks(mapping):
    """
    Create a new dataframe with many assets and a new column indicating
    the asset that row's data belongs to.

    Parameters:
        - mapping: A key-value mapping of the form {asset_name: asset_df}

    Returns:
        A new `pandas.DataFrame` object
    """
    group_df = pd.DataFrame()

    for stock, stock_data in mapping.items():
        df = stock_data.copy(deep=True)
        df['name'] = stock
        group_df = group_df.append(df, sort=True)

    group_df.index = pd.to_datetime(group_df.index)

    return group_df


@validate_df(columns={'name'}, instance_method=False)
def describe_group(data):
    """
    Run `describe()` on the asset group created with `group_stocks()`.

    Parameters:
        - data: The group data resulting from `group_stocks()`

    Returns:
        The transpose of the grouped description statistics.
    """
    return data.groupby('name').describe().T


@validate_df(columns=set(), instance_method=False)
def make_portfolio(data, date_level='date'):
    """
    Make a portfolio of assets by grouping by date and summing all columns.

    Note: the caller is responsible for making sure the dates line up across
    assets and handling when they don't.
    """
    return data.groupby(level=date_level).sum()
