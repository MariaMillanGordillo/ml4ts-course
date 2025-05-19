import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series

__all__ = [
    "ExampleDataLoader",
    "bootstrapped_forecasts",
    "plot_forecast_with_intervals",
    "plot_residuals",
    "plot_series_slice",
]


_DATA_PATH = Path(__file__).parents[1] / "data"


class ExampleDataLoader:
    """
    A simple data loader for example datasets related to chemical processes and energy demand.

    Attributes:
        name (str): The default dataset name to load. Must be one of the keys in `_data`.

    Example:
        loader = ExampleDataLoader("process")
        df = loader.load()  # Loads the full DataFrame for the "process" dataset.
        series = loader.load(name="temperature", idx=5)  # Loads a specific time series.
    """

    _data = {
        "process": _DATA_PATH / "chemical_process.parquet",
        "energy": _DATA_PATH / "energy_demand.parquet",
    }

    def __init__(self, name: str):
        self.name = name

    @t.overload
    def load(self) -> pd.DataFrame: ...

    @t.overload
    def load(self, *, name: str, idx: int = ...) -> pd.Series: ...
    def load(self, *, name: t.Optional[str] = None, idx: t.Optional[int] = None) -> pd.DataFrame | pd.Series:
        if not name:
            return pd.read_parquet(self._data[self.name])

        series = self.load().loc[idx or 1, name].reset_index(drop=True)
        series.name = name
        return series


def plot_series_slice(
    series: pd.Series,
    /,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp,
    freq: str,
    return_data: bool = False,
    **kwargs,
) -> pd.Series | None:
    """
    Plot a slice of a time series between `start` and `stop` with a given frequency.

    Parameters:
        series (pd.Series): The input time series with a DateTimeIndex.
        start (str or pd.Timestamp): Start time of the window.
        stop (str or pd.Timestamp): Stop time of the window.
        freq (str): Frequency string for the date range (e.g., 'D', 'H').
        return_data (bool): Whether to return the sliced series.
        **kwargs: Additional keyword arguments passed to `plot_series`.

    Returns:
        Optional[pd.Series]: The sliced time series if `return_data=True`, else None.
    """
    window = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(stop), freq=freq)
    sliced = series.loc[series.index.isin(window)]

    plot_series(sliced, **kwargs)

    if return_data:
        return sliced


def plot_residuals(residuals: pd.Series, *, show: bool = True, return_fig: bool = False) -> t.Optional[plt.Figure]:
    """
    Plot a histogram and Q-Q plot to visualize the distribution of residuals.

    Parameters:
        residuals (pd.Series): Residual values to be analyzed.
        show (bool): Whether to display the plots with plt.show().
        return_fig (bool): If True, returns the matplotlib Figure object.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure, if `return_fig=True`, else None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Histogram with fitted normal distribution
    ax1 = axes[0]
    ax1.hist(residuals, bins="auto", density=True, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Residuals")
    ax1.set_ylabel("Density")
    ax1.set_title("Histogram of Residuals")
    ax1.grid(True, linestyle="--", alpha=0.4)

    x = np.linspace(residuals.min(), residuals.max(), 200)
    pdf = stats.norm.pdf(x, loc=residuals.mean(), scale=residuals.std())
    ax1.plot(x, pdf, "r-", lw=2, label="Normal PDF")
    ax1.legend()

    # Q-Q plot
    ax2 = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot of Residuals")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if show:
        plt.show()

    return fig if return_fig else None


def bootstrapped_forecasts(
    forecaster,
    y: pd.Series,
    fh: t.Union[np.ndarray, pd.Index, ForecastingHorizon],
    n_bootstraps: int = 100,
    random_state: t.Union[int, None] = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate bootstrapped prediction intervals using residual resampling.

    Parameters:
        forecaster: A fitted sktime forecaster.
        y (pd.Series): The observed time series used for residual computation.
        fh (array-like or ForecastingHorizon): Forecasting horizon.
        n_bootstraps (int): Number of bootstrap samples.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame of shape (len(fh), n_bootstraps), containing the bootstrapped forecasts.
    """
    rng = np.random.default_rng(random_state)

    # Forecast point prediction
    y_pred = forecaster.predict(fh)

    # Compute residuals using in-sample prediction
    in_sample_fh = ForecastingHorizon(y.index, is_relative=False)
    y_fit = forecaster.predict(in_sample_fh)
    residuals = (y - y_fit).dropna()
    residuals_array = residuals.to_numpy()

    # Ensure fh is sized
    fh_array = np.asarray(fh)
    fh_len = len(fh_array)

    # Prepare bootstrapped forecast container
    bootstrapped_forecasts = pd.DataFrame(index=y_pred.index, columns=range(n_bootstraps), dtype=float)

    for i in range(n_bootstraps):
        resampled = rng.choice(residuals_array, size=fh_len, replace=True)
        bootstrapped_forecasts.iloc[:, i] = y_pred + resampled

    return y_pred, bootstrapped_forecasts


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    """Ensures the index of a Series is a DatetimeIndex."""
    if isinstance(series.index, pd.PeriodIndex):
        return pd.Series(series.values, index=series.index.to_timestamp())
    return series


def plot_forecast_with_intervals(
    *,
    actual: pd.Series,
    forecast: pd.Series,
    lower: t.Optional[pd.Series] = None,
    upper: t.Optional[pd.Series] = None,
    interval_label: str = "Prediction Interval (95%)",
    figsize: tuple[int, int] = (10, 5),
    title: str = "Forecast with Prediction Interval",
    xlabel: str = "Time",
    ylabel: str = "Value",
    actual_style: t.Optional[dict] = None,
    forecast_style: t.Optional[dict] = None,
    interval_style: t.Optional[dict] = None,
    fill_alpha: t.Optional[float] = 0.1,
    show: bool = True,
) -> None:
    """
    Plot time series forecast with optional prediction intervals.

    Parameters:
        actual (pd.Series): Historical data.
        forecast (pd.Series): Forecasted values.
        lower (pd.Series, optional): Lower prediction bound.
        upper (pd.Series, optional): Upper prediction bound.
        interval_label (str): Label for prediction interval fill.
        figsize (tuple): Size of the figure.
        title (str): Plot title.
        xlabel (str): Label for X axis.
        ylabel (str): Label for Y axis.
        actual_style (dict): Custom style for actual line.
        forecast_style (dict): Custom style for forecast line.
        interval_style (dict): Style for upper/lower bound lines.
        fill_alpha (float): Transparency of the fill between bounds.
        show (bool): Whether to display the plot.
    """
    actual_style = actual_style or {"color": "black", "label": "Historical"}
    forecast_style = forecast_style or {"color": "blue", "label": "Forecast"}
    interval_style = interval_style or {"color": "red", "linestyle": "--", "alpha": 0.4}

    plt.figure(figsize=figsize)

    actual_ts = _ensure_datetime_index(actual)
    plt.plot(actual_ts.index, actual_ts, **actual_style)

    forecast_ts = _ensure_datetime_index(forecast)
    plt.plot(forecast_ts.index, forecast_ts, **forecast_style)

    if lower is not None and upper is not None:
        lower_ts = _ensure_datetime_index(lower)
        upper_ts = _ensure_datetime_index(upper)
        plt.plot(lower_ts.index, lower_ts.to_numpy(), label="Lower Bound", **interval_style)
        plt.plot(upper_ts.index, upper_ts.to_numpy(), label="Upper Bound", **interval_style)
        plt.fill_between(
            lower_ts.index,
            lower_ts.to_numpy(),
            upper_ts.to_numpy(),
            color=interval_style.get("color", "red"),
            alpha=fill_alpha,
            label=interval_label,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()
