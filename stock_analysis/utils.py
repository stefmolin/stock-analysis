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

def label_sanitizer(method, *args, **kwargs):
    """
    Decorator around a method that returns a dataframe to
    clean up all labels in said dataframe (column names and index name)
    by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters:
        - method: The dataframe with labels you want to fix.
        - args: Additional positional arguments to pass to the wrapped method.
        - kwargs: Additional keyword arguments to pass to the wrapped method.

    Returns:
        A decorated method.
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

def group_stocks(mapping):
    """
    Create a new dataframe with many assets and a new column indicating
    the asset that row's data belongs to.

    Parameters:
        - mapping: A key-value mapping of the form { asset_name : asset_df }

    Returns:
        A new pandas DataFrame
    """
    group_df = pd.DataFrame()

    for stock, stock_data in mapping.items():
        df = stock_data.copy(deep=True)
        df['name'] = stock
        group_df = group_df.append(df, sort=True)

    group_df.index = pd.to_datetime(group_df.index)

    return group_df

def describe_group(data):
    """
    Run `describe()` on the asset group created with `group_stocks()`.

    Parameters:
        - data: The group data resulting from `group_stocks()`

    Returns:
        The transpose of the grouped description statistics.
    """
    return data.groupby('name').describe().T
