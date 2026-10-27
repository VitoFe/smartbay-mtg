"""Addendum 65's weather windows on a synthetic station: the past reads only hours up to issuance, the
future the hours after it, and precipitation is summed where the other channels are averaged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import v2_weather_oracle as W  # noqa: E402


def synthetic(monkeypatch):
    index = pd.date_range('2025-01-01', periods=100, freq='h', tz='UTC')
    frame = pd.DataFrame({c: np.arange(100, dtype=float) for c in W.CHANNELS}, index=index)
    monkeypatch.setattr(W, 'station', lambda: frame)
    return index


def test_past_weather_reads_the_six_hours_ending_at_issuance(monkeypatch):
    index = synthetic(monkeypatch)
    past = W.past_weather(index[[10]])[0]
    precip = list(W.CHANNELS).index('precip')
    others = [i for i in range(len(W.CHANNELS)) if i != precip]
    assert np.allclose(past[others], np.mean(np.arange(5, 11)))
    assert past[precip] == np.sum(np.arange(5, 11))


def test_future_weather_reads_after_issuance_to_the_target(monkeypatch):
    index = synthetic(monkeypatch)
    out = W.future_weather(index[[10]], 3)[0]
    n = len(W.CHANNELS)
    window, value = out[:n], out[n:]
    precip = list(W.CHANNELS).index('precip')
    others = [i for i in range(n) if i != precip]
    assert np.allclose(window[others], 12.0)          # hours 11, 12, 13
    assert window[precip] == 36.0
    assert np.allclose(value, 13.0)                   # the target hour


def test_future_weather_never_reads_the_issuance_hour(monkeypatch):
    index = synthetic(monkeypatch)
    out = W.future_weather(index[[10]], 1)[0]
    assert np.allclose(out, 11.0)
