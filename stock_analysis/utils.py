"""Utility functions for stock analysis."""

from functools import wraps
import re

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
