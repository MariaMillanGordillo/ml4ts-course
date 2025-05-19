import typing as t
from pathlib import Path

import pandas as pd
from sktime.utils.plotting import plot_series

__all__ = ["ExampleDataLoader", "plot_series_slice"]


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
