"""Visualize financial instruments."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import validate_df

class Visualizer:
    """Base visualizer class not intended for direct use."""

    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df):
        """Visualizer has a pandas dataframe as an attribute."""
        self.data = df

    @staticmethod
    def add_reference_line(ax, x=None, y=None, **kwargs):
        """
        Static method for adding reference lines to plots.

        Parameters:
            - ax: The matplotlib Axes object to add the reference line to.
            - x, y: The x, y value to draw the line at as a
                    single value or numpy array-like structure.
                        - For horizontal: pass only `y`
                        - For vertical: pass only `x`
                        - For AB line: pass both `x` and `y`
                        for all coordinates on the line
            - kwargs: Additional keyword arguments to pass to the plotting
                      function.

        Returns:
            The Axes object passed in.
        """
        try:
            # in case numpy array-like structures are passed -> AB line
            if x.shape and y.shape:
                ax.plot(x, y, **kwargs)
        except:
            # error triggers if at least one isn't a numpy array-like structure
            try:
                if not x and not y:
                    raise ValueError(
                        'You must provide an `x` or a `y` at a minimum!'
                    )
                elif x and not y:
                    # vertical line
                    ax.axvline(x, **kwargs)
                elif not x and y:
                    # horizontal line
                    ax.axhline(y, **kwargs)
            except:
                raise ValueError(
                    'If providing only `x` or `y`, it must be a single value'
                )
        ax.legend()
        return ax

    @staticmethod
    def shade_region(ax, x=tuple(), y=tuple(), **kwargs):
        """
        Static method for shading a region on a plot.

        Parameters:
            - ax: The matplotlib Axes object to add the shaded region to.
            - x: Tuple with the `xmin` and `xmax` bounds for the rectangle
                 drawn vertically.
            - y: Tuple with the `ymin` and `ymax` bounds for the rectangle
                 drawn vertically.
            - kwargs: Additional keyword arguments to pass to the plotting
                      function.

        Returns:
            The Axes object passed in.
        """
        if not x and not y:
            raise ValueError(
                'You must provide an x or a y min/max tuple at a minimum!'
            )
        elif x and y:
            raise ValueError('You can only provide `x` or `y`.')
        elif x and not y:
            # vertical span
            ax.axvspan(*x, **kwargs)
        elif not x and y:
            # horizontal span
            ax.axhspan(*y, **kwargs)
        return ax

    @staticmethod
    def _iter_handler(items):
        """
        Static method for making a list out of a item if it isn't a list or
        tuple already.

        Parameters:
            - items: The variable to make sure it is a list.

        Returns:
            The input as a list or tuple.
        """
        if not isinstance(items, (list, tuple)):
            items = [items]
        return items

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        To be implemented by subclasses. Defines how to add lines resulting
        from window calculations.
        """
        raise NotImplementedError('To be implemented by subclasses!')

    def moving_average(self, column, periods, **kwargs):
        """
        Add line(s) for the moving average of a column.

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule or list of rules for resampling,
                       like '20D' for 20-day periods.
            - kwargs: Additional arguments to pass down to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        return self._window_calc_func(
            column, periods, name='MA',
            func=pd.DataFrame.resample, named_arg='rule', **kwargs
        )

    def exp_smoothing(self, column, periods, **kwargs):
        """
        Add line(s) for the exponentially smoothed moving average of a column.

        Parameters:
            - column: The name of the column to plot.
            - periods: The span or list of spans for smoothing,
                       like 20 for 20-day periods.
            - kwargs: Additional arguments to pass down to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        return self._window_calc_func(
            column, periods, name='EWMA',
            func=pd.DataFrame.ewm, named_arg='span', **kwargs
        )

    # abstract methods for subclasses to define
    def evolution_over_time(self, column, **kwargs):
        """To be implemented by subclasses for generating line plots."""
        raise NotImplementedError('To be implemented by subclasses!')

    def boxplot(self, **kwargs):
        """To be implemented by subclasses for generating boxplots."""
        raise NotImplementedError('To be implemented by subclasses!')

    def histogram(self, column, **kwargs):
        """To be implemented by subclasses for generating histograms."""
        raise NotImplementedError('To be implemented by subclasses!')

    def after_hours_trades(self):
        """To be implemented by subclasses."""
        raise NotImplementedError('To be implemented by subclasses!')

    def pairplot(self, **kwargs):
        """To be implemented by subclasses for generating pairplots."""
        raise NotImplementedError('To be implemented by subclasses!')

class StockVisualizer(Visualizer):
    """Visualizer for a single stock."""

    def evolution_over_time(self, column, **kwargs):
        """
        Visualize the evolution over time of a column.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        return self.data.plot.line(y=column, **kwargs)

    def boxplot(self, **kwargs):
        """
        Generate boxplots for all columns.

        Parameters:
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        return self.data.plot(kind='box', **kwargs)

    def histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        return self.data.plot.hist(y=column, **kwargs)

    def trade_volume(self, tight=False, **kwargs):
        """
        Visualize the trade volume and closing price.

        Parameters:
            - tight: Whether or not to attempt to match up the resampled
                     bar plot on the bottom to the line plot on the top.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib Figure object.
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 15))
        self.data.close.plot(ax=axes[0], title='Closing Price')
        monthly = self.data.volume.resample('1M').sum()
        monthly.index = monthly.index.strftime('%b\n%Y')
        monthly.plot(
            kind='bar', ax=axes[1], color='blue', rot=0, title='Volume Traded'
        )
        if tight:
            axes[0].set_xlim(self.data.index.min(), self.data.index.max())
            axes[1].set_xlim(-0.25, axes[1].get_xlim()[1] - 0.25)
        plt.close()
        return fig

    def after_hours_trades(self):
        """
        Visualize the effect of after hours trading on this asset.

        Returns:
            A matplotlib Figure object.
        """
        after_hours = (self.data.open - self.data.close.shift())

        monthly_effect = after_hours.resample('1M').sum()
        fig, axes = plt.subplots(1, 2, figsize=(15, 3))

        after_hours.plot(
            ax=axes[0],
            title='After hours trading\n(Open Price - Prior Day\'s Close)'
        )

        monthly_effect.index = monthly_effect.index.strftime('%b')
        monthly_effect.plot(
            ax=axes[1],
            kind='bar',
            title='After hours trading monthly effect',
            color=np.where(monthly_effect >= 0, 'g', 'r'),
            rot=90
        ).axhline(0, color='black', linewidth=1)
        plt.close()
        return fig

    def open_to_close(self, figsize=(10, 4)):
        """
        Visualize the daily change from open to close price.

        Parameters:
            - figsize: A tuple of (width, height) for the plot dimensions.

        Returns:
            A matplotlib plot object.
        """
        is_higher = self.data.close - self.data.open > 0

        fig = plt.figure(figsize=figsize)

        for exclude_mask, color, label in zip(
            (is_higher, np.invert(is_higher)),
            ('g', 'r'),
            ('price rose', 'price fell')
        ):
            plt.fill_between(
                self.data.index,
                self.data.open,
                self.data.close,
                figure=fig,
                where=exclude_mask,
                color=color,
                label=label
            )
        plt.suptitle('Daily price change (open to close)')
        plt.legend()
        plt.close()
        return fig

    def fill_between_other(self, other_df, figsize=(10, 4)):
        """
        Visualize the difference in closing price between assets.

        Parameters:
            - other_df: The dataframe with the other asset's data.
            - figsize: A tuple of (width, height) for the plot dimensions.

        Returns:
            A matplotlib Figure object.
        """
        is_higher = self.data.close - other_df.close > 0

        fig = plt.figure(figsize=figsize)

        for exclude_mask, color, label in zip(
            (is_higher, np.invert(is_higher)),
            ('g', 'r'),
            ('asset is higher', 'asset is lower')
        ):
            plt.fill_between(
                self.data.index,
                self.data.close,
                other_df.close,
                figure=fig,
                where=exclude_mask,
                color=color,
                label=label
            )
        plt.suptitle(
            'Differential between asset closing price (this - other)'
        )
        plt.legend()
        plt.close()
        return fig

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines using
        a window calculation.

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                       resampling/smoothing function, like '20D' for 20-day periods
                       (for resampling) or 20 for a 20-day span (smoothing)
            - name: The name of the window calculation (to show in the legend).
            - func: The window calculation function.
            - named_arg: The name of the argument `periods` is being passed as.
            - kwargs: Additional arguments to pass down to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        ax = self.data.plot(y=column, **kwargs)
        for period in self._iter_handler(periods):
            self.data[column].pipe(
                func, **{named_arg: period}
            ).mean().plot(
                ax=ax,
                linestyle='--',
                label=f'{period if isinstance(period, str) else str(period) + "D"} {name}'
            )
        plt.legend()
        return ax

    def correlation_heatmap(self, other):
        """
        Plot the correlations between the same column between this asset and
        another one with a heatmap.

        Parameters:
            - other: The other dataframe.

        Returns:
            A seaborn heatmap
        """
        corrs = self.data.corrwith(other)
        corrs = corrs[~pd.isnull(corrs)]
        size = len(corrs)
        matrix = np.zeros((size, size), float)
        for i, corr in zip(range(size), corrs):
            matrix[i][i] = corr

        # create mask to only show diagonal
        mask = np.ones_like(matrix)
        np.fill_diagonal(mask, 0)

        return sns.heatmap(
            matrix,
            annot=True,
            xticklabels=self.data.columns,
            yticklabels=self.data.columns,
            center=0,
            mask=mask
        )

    def pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset.

        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(self.data, **kwargs)

    def jointplot(self, other, column, **kwargs):
        """
        Generate a seaborn jointplot for given column in asset compared to
        another asset.

        Parameters:
            - other: The other asset's dataframe
            - column: The column name to use for the comparison.
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn jointplot
        """
        return sns.jointplot(
            x=self.data[column],
            y=other[column],
            **kwargs
        )

class AssetGroupVisualizer(Visualizer):
    """Class for visualizing groups of assets in a single dataframe."""

    # override for group visuals
    def __init__(self, df, group_by='name'):
        """This object keeps track of which column it needs to group by."""
        super().__init__(df)
        self.group_by = group_by

    def evolution_over_time(self, column, **kwargs):
        """
        Visualize the evolution over time of a column for all assets in group.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        if 'ax' not in kwargs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        else:
            ax = kwargs.pop('ax')
        return sns.lineplot(
            x=self.data.index,
            y=column,
            hue=self.group_by,
            data=self.data,
            ax=ax,
            **kwargs
        )

    def boxplot(self, column, **kwargs):
        """
        Generate boxplots for a given column in all assets.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        return sns.boxplot(
            x=self.group_by,
            y=column,
            data=self.data,
            **kwargs
        )

    def _get_layout(self):
        """
        Helper method for getting an autolayout of subplots (1 per group).

        Returns:
            The matplotlib Figure and Axes objects to plot with.
        """
        subplots_needed = self.data[self.group_by].nunique()
        rows = math.ceil(subplots_needed / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
        if rows > 1:
            axes = axes.flatten()
        if subplots_needed < len(axes):
            # remove excess axes from autolayout
            for i in range(subplots_needed, len(axes)):
                # can't use comprehension here
                fig.delaxes(axes[i])
        return fig, axes

    def histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column for all assets in group.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        fig, axes = self._get_layout()
        for ax, (name, data) in zip(axes, self.data.groupby(self.group_by)):
            sns.distplot(data[column], ax=ax, axlabel=f'{name} - {column}')
        plt.close()
        return fig

    def _window_calc_func(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines using
        a window calculation.

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                       resampling/smoothing function, like '20D' for 20-day periods
                       (for resampling) or 20 for a 20-day span (smoothing)
            - name: The name of the window calculation (to show in the legend).
            - func: The window calculation function.
            - named_arg: The name of the argument `periods` is being passed as.
            - kwargs: Additional arguments to pass down to the plotting function.

        Returns:
            A matplotlib Axes object.
        """
        fig, axes = self._get_layout()
        for ax, asset_name in zip(axes, self.data[self.group_by].unique()):
            subset = self.data[self.data[self.group_by] == asset_name]
            ax = subset.plot(y=column, ax=ax, label=asset_name, **kwargs)
            for period in self._iter_handler(periods):
                subset[column].pipe(
                    func, **{named_arg: period}
                ).mean().plot(
                    ax=ax,
                    linestyle='--',
                    label=f'{period if isinstance(period, str) else str(period) + "D"} {name}'
                )
            ax.legend()
        return ax

    def after_hours_trades(self):
        """
        Visualize the effect of after hours trading on this asset.

        Returns:
            A matplotlib Figure object.
        """
        num_categories = self.data[self.group_by].nunique()
        fig, axes = plt.subplots(
            num_categories,
            2,
            figsize=(15, 8*num_categories)
        )

        for ax, (name, data) in zip(axes, self.data.groupby(self.group_by)):
            after_hours = (data.open - data.close.shift())

            monthly_effect = after_hours.resample('1M').sum()

            after_hours.plot(
                ax=ax[0],
                title=f'{name} Open Price - Prior Day\'s Close'
            )

            monthly_effect.index = monthly_effect.index.strftime('%b')
            monthly_effect.plot(
                ax=ax[1],
                kind='bar',
                title=f'{name} after hours trading monthly effect',
                color=np.where(monthly_effect >= 0, 'g', 'r'),
                rot=90
            ).axhline(0, color='black', linewidth=1)
        plt.close()
        return fig

    def pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset group.

        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(
            self.data.pivot_table(
                values='close', index=self.data.index, columns='name'
            ),
            diag_kind='kde',
            **kwargs
        )

    def heatmap(self, pct_change=False, **kwargs):
        """
        Generate a seaborn heatmap for correlations between assets.

        Parameters:
            - pct_change: Whether or not to show the correlations of the
                          daily percent change in price or just use
                          the closing price.
            - kwargs: Keyword arguments to pass down to `sns.heatmap()`

        Returns:
            A seaborn heatmap
        """
        pivot = self.data.pivot_table(
            values='close', index=self.data.index, columns='name'
        )
        if pct_change:
            pivot = pivot.pct_change()
        return sns.heatmap(pivot.corr(), annot=True, center=0, **kwargs)
