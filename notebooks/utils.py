import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as special
import scipy.stats as stats
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.transformations.base import BaseTransformer
from sktime.utils.plotting import plot_series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

__all__ = [
    "ExampleDataLoader",
    "ArmaProcess",
    "BootstrappedForecaster",
    "MyBoxCoxTransformer",
    "MyMonthlyAdjuster",
    "plot_forecast_with_intervals",
    "plot_residuals",
    "plot_series_slice",
    "plot_acf",
    "plot_pacf",
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
        "monthly_demand": _DATA_PATH / "electricity_au_month.parquet",
        "electricity": _DATA_PATH / "electricity_au.parquet",
        "google": _DATA_PATH / "google.parquet",
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


def plot_residuals(
    residuals: pd.Series, *, show: bool = True, return_fig: bool = False, figsize: tuple = (12, 4)
) -> t.Optional[plt.Figure]:
    """
    Plot a histogram and Q-Q plot to visualize the distribution of residuals.

    Parameters:
        residuals (pd.Series): Residual values to be analyzed.
        show (bool): Whether to display the plots with plt.show().
        return_fig (bool): If True, returns the matplotlib Figure object.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure, if `return_fig=True`, else None.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

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


def plot_decomposition(y, *, trend, seasonality, residual, model="additive"):
    """
    Plot time series decomposition with trend, seasonality, and residual components.

    Parameters:
        y (pd.Series): Original time series data.
        trend (pd.Series): Trend component of the time series.
        seasonality (pd.Series): Seasonal component of the time series.
        residual (pd.Series): Residual component of the time series.
        model (str, optional): Type of decomposition, either "additive" or "multiplicative". Defaults to "additive".

    Returns:
        None
    """
    fig, ax = plot_series(y, labels=["Original series"])  # type: ignore

    if model == "additive":
        reconstructed = trend + seasonality
        label = "Trend + Seasonality"
    elif model == "multiplicative":
        reconstructed = trend * seasonality
        label = "Trend × Seasonality"
    else:
        raise ValueError("model must be either 'additive' or 'multiplicative'")

    ax.plot(reconstructed, color="purple", linestyle="-.", label=label)  # type: ignore
    ax.plot(trend, color="red", linestyle="--", alpha=0.6, label="Trend")  # type: ignore
    ax.plot(residual, color="black", linestyle=":", alpha=0.6, label="Residual")  # type: ignore

    plt.legend()
    plt.tight_layout()
    plt.show()

    return fig, ax


class BootstrappedForecaster:
    def __init__(self, forecaster: BaseForecaster, random_state: int = 100):
        self._forecaster = forecaster
        self._rng = np.random.default_rng(random_state)

    def predict(
        self,
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
        y_pred: pd.Series = self._forecaster.predict(fh)  # type: ignore

        # Compute residuals using in-sample prediction
        residuals = (y - self._forecaster.predict(ForecastingHorizon(y.index, is_relative=False))).dropna().to_numpy()

        # Prepare bootstrapped forecast container
        forecasts = pd.DataFrame(index=y_pred.index, columns=range(n_bootstraps), dtype=float)

        for i in range(n_bootstraps):
            resampled = self._rng.choice(residuals, size=len(np.asarray(fh)), replace=True)
            forecasts.iloc[:, i] = y_pred + resampled

        return y_pred, forecasts


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
    figsize: tuple[int, int] = (12, 4),
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


def plot_series_and_acf(
    series: pd.Series,
    *,
    title: str = "Time series",
    lags: int = 40,
    alpha: float = 0.05,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
    return_fig: bool = False,
) -> t.Optional[plt.Figure]:
    """
    Plot a time series and its autocorrelation function.

    Parameters:
        series (pd.Series): Time series data to be plotted.
        title (str): Title for the time series plot.
        lags (int): Number of lags to show in ACF plot.
        alpha (float): Confidence level for the ACF.
        figsize (Tuple[int, int]): Size of the full figure (width, height).
        show (bool): Whether to display the plot using plt.show().
        return_fig (bool): If True, returns the matplotlib Figure object.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure, if `return_fig=True`.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].plot(series.index, series, marker="o", markersize=2)
    axes[0].set_title(title)
    axes[0].set_ylabel("Value")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    plot_acf(series, lags=lags, alpha=alpha, ax=axes[1], zero=False, auto_ylims=True)

    if show:
        plt.show()

    return fig if return_fig else None


def plot_series_and_pacf(
    series: pd.Series,
    *,
    title: str = "Time series",
    lags: int = 40,
    alpha: float = 0.05,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
    return_fig: bool = False,
) -> t.Optional[plt.Figure]:
    """
    Plot a time series and its partial-autocorrelation function.

    Parameters:
        series (pd.Series): Time series data to be plotted.
        title (str): Title for the time series plot.
        lags (int): Number of lags to show in ACF plot.
        alpha (float): Confidence level for the ACF.
        figsize (Tuple[int, int]): Size of the full figure (width, height).
        show (bool): Whether to display the plot using plt.show().
        return_fig (bool): If True, returns the matplotlib Figure object.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure, if `return_fig=True`.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].plot(series.index, series, marker="o", markersize=2)
    axes[0].set_title(title)
    axes[0].set_ylabel("Value")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    plot_pacf(series, lags=lags, alpha=alpha, ax=axes[1], zero=False, auto_ylims=True)

    if show:
        plt.show()

    return fig if return_fig else None


def plot_acf_and_pacf(
    series: pd.Series,
    *,
    lags: int = 40,
    alpha: float = 0.05,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
    return_fig: bool = False,
) -> t.Optional[plt.Figure]:
    """
    Plot ACF and PACF functions.

    Parameters:
        series (pd.Series): Time series data to be plotted.
        title (str): Title for the time series plot.
        lags (int): Number of lags to show in ACF plot.
        alpha (float): Confidence level for the ACF.
        figsize (Tuple[int, int]): Size of the full figure (width, height).
        show (bool): Whether to display the plot using plt.show().
        return_fig (bool): If True, returns the matplotlib Figure object.

    Returns:
        Optional[plt.Figure]: The matplotlib Figure, if `return_fig=True`.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    plot_acf(series, lags=lags, alpha=alpha, ax=axes[0], zero=False, auto_ylims=True)
    plot_pacf(series, lags=lags, alpha=alpha, ax=axes[1], zero=False, auto_ylims=True)

    if show:
        plt.show()

    return fig if return_fig else None


class MyBoxCoxTransformer:
    def __init__(self, alpha: float | None = None):
        self.alpha = alpha
        self._is_fit = False

    def fit(self, y):
        _, alpha_ = stats.boxcox(y)  # type: ignore
        self.alpha = alpha_
        self._is_fit = True
        return self

    def transform(self, y):
        if (not self._is_fit) and (self.alpha is None):
            raise ValueError("Alpha value is None and transformer is not fit")
        return stats.boxcox(y, self.alpha)

    def inverse_transform(self, y):
        if (not self._is_fit) and (self.alpha is None):
            raise ValueError("Alpha value is None and transformer is not fit")

        return special.inv_boxcox(y, self.alpha)

    def fit_transform(self, y):
        return self.fit(y).transform(y)


class MyMonthlyAdjuster(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.avg_days_per_month = 365.25 / 12

    def _fit(self, X, y=None):
        # No fitting required for this transformer
        return self

    def _transform(self, X, y=None):
        X = X.copy()

        input_is_series = isinstance(X, pd.Series)
        if input_is_series:
            X = X.to_frame(name="__value__")

        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        X["Date"] = X.index
        X["__DaysInMonth__"] = X["Date"].dt.days_in_month

        adjusted_cols = {}
        for col in X.columns.difference(["Date", "__DaysInMonth__"]):
            adjusted_col = (X[col] / X["__DaysInMonth__"]) * self.avg_days_per_month
            adjusted_cols[col] = adjusted_col

        adjusted = pd.DataFrame(adjusted_cols, index=X["Date"])  # type: ignore

        if input_is_series:
            return adjusted.squeeze()

        return adjusted
