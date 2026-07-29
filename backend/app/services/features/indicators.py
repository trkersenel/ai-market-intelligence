"""Technical indicators as pure functions.

No I/O, no database, no pandas. Every function takes sequences of floats and
returns a list of the same length, with ``None`` for the leading sessions where
the indicator is not yet defined. That alignment property is the contract the
whole module rests on: the caller can always ``zip`` a result straight back onto
its dates without tracking offsets, and a warm-up period can never be silently
mistaken for a real value.

Two deliberate choices, both of which are common sources of wrong numbers:

**Wilder's smoothing** is used for RSI and ATR, not a simple moving average.
Wilder defined both indicators with a specific recursive smoothing, and every
charting platform implements them that way. Substituting an SMA produces values
that look plausible, track the right direction, and disagree with every other
tool a user might check against.

**Floats, not Decimal.** Prices are stored as ``NUMERIC`` because money must be
exact. Indicators are different: they are statistics over prices -- means,
standard deviations, exponential decay -- and are inherently approximate.
Computing an EMA in ``Decimal`` would be slower, would still need rounding at
every step, and would imply a precision the indicator does not have. Conversion
back to ``Decimal`` happens once, at the storage boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Trading sessions in a year, the standard annualisation factor for daily data.
TRADING_DAYS_PER_YEAR = 252

#: Sessions per calendar week and month, in trading days rather than calendar
#: days. A "weekly return" over 7 calendar days would silently span a variable
#: number of sessions depending on holidays.
SESSIONS_PER_WEEK = 5
SESSIONS_PER_MONTH = 21


@dataclass(frozen=True)
class MacdResult:
    """MACD line, its signal line, and the histogram between them."""

    macd: list[float | None]
    signal: list[float | None]
    histogram: list[float | None]


@dataclass(frozen=True)
class BollingerResult:
    """The three Bollinger bands."""

    upper: list[float | None]
    middle: list[float | None]
    lower: list[float | None]


def simple_moving_average(values: Sequence[float], period: int) -> list[float | None]:
    """Return the rolling arithmetic mean over ``period`` observations.

    Args:
        values: Ordered observations, oldest first.
        period: Window length.

    Returns:
        A list aligned to ``values``; the first ``period - 1`` entries are
        ``None``.

    Raises:
        ValueError: If ``period`` is not positive.
    """
    _require_positive_period(period)
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result

    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period
    for index in range(period, len(values)):
        # Rolling update rather than re-summing the window: O(n) instead of
        # O(n*period), which matters when backfilling 200-session averages.
        window_sum += values[index] - values[index - period]
        result[index] = window_sum / period
    return result


def exponential_moving_average(values: Sequence[float], period: int) -> list[float | None]:
    """Return the exponentially weighted moving average.

    Seeded with the simple average of the first ``period`` observations, which
    is the convention every charting platform uses. Seeding with the first value
    instead would make early output depend heavily on one arbitrary price.

    Args:
        values: Ordered observations, oldest first.
        period: Span controlling the decay, ``alpha = 2 / (period + 1)``.

    Returns:
        A list aligned to ``values``, ``None`` before the seed.
    """
    _require_positive_period(period)
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result

    multiplier = 2.0 / (period + 1)
    previous = sum(values[:period]) / period
    result[period - 1] = previous
    for index in range(period, len(values)):
        previous = (values[index] - previous) * multiplier + previous
        result[index] = previous
    return result


def relative_strength_index(closes: Sequence[float], period: int = 14) -> list[float | None]:
    """Return Wilder's RSI, bounded to 0-100.

    Args:
        closes: Closing prices, oldest first.
        period: Lookback, conventionally 14.

    Returns:
        A list aligned to ``closes``; the first ``period`` entries are ``None``
        because RSI needs ``period`` price *changes*, one more than prices.

    Notes:
        A window with no losses gives an undefined ratio; RSI is 100 by
        definition there (and 0 for no gains), rather than a division error.
    """
    _require_positive_period(period)
    result: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return result

    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, len(closes)):
        change = closes[index] - closes[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    result[period] = _rsi_from_averages(average_gain, average_loss)

    for index in range(period, len(gains)):
        # Wilder's smoothing: a recursive average with weight 1/period on the
        # newest observation. This is *not* an SMA of the last `period` changes.
        average_gain = (average_gain * (period - 1) + gains[index]) / period
        average_loss = (average_loss * (period - 1) + losses[index]) / period
        result[index + 1] = _rsi_from_averages(average_gain, average_loss)
    return result


def macd(
    closes: Sequence[float],
    *,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MacdResult:
    """Return the MACD line, signal line and histogram.

    Args:
        closes: Closing prices, oldest first.
        fast_period: Span of the fast EMA.
        slow_period: Span of the slow EMA.
        signal_period: Span of the EMA applied to the MACD line.

    Returns:
        Three lists aligned to ``closes``.

    Raises:
        ValueError: If ``fast_period`` is not shorter than ``slow_period``.
    """
    if fast_period >= slow_period:
        msg = "fast_period must be shorter than slow_period"
        raise ValueError(msg)

    fast = exponential_moving_average(closes, fast_period)
    slow = exponential_moving_average(closes, slow_period)

    macd_line: list[float | None] = [
        None if f is None or s is None else f - s for f, s in zip(fast, slow, strict=True)
    ]

    # The signal EMA is defined only over the sessions where MACD itself exists,
    # so the defined stretch is extracted, smoothed, and placed back at its
    # original offsets. Feeding the None-padded list to the EMA would treat the
    # warm-up as data.
    defined = [(index, value) for index, value in enumerate(macd_line) if value is not None]
    signal_line: list[float | None] = [None] * len(closes)
    if defined:
        smoothed = exponential_moving_average([value for _, value in defined], signal_period)
        for (original_index, _), value in zip(defined, smoothed, strict=True):
            signal_line[original_index] = value

    histogram: list[float | None] = [
        None if m is None or s is None else m - s
        for m, s in zip(macd_line, signal_line, strict=True)
    ]
    return MacdResult(macd=macd_line, signal=signal_line, histogram=histogram)


def bollinger_bands(
    closes: Sequence[float], *, period: int = 20, num_std: float = 2.0
) -> BollingerResult:
    """Return Bollinger bands around a simple moving average.

    Args:
        closes: Closing prices, oldest first.
        period: Window for the middle band and the deviation.
        num_std: Band width in standard deviations.

    Returns:
        Upper, middle and lower bands aligned to ``closes``.

    Notes:
        Uses the *population* standard deviation, dividing by ``n``. Bollinger's
        definition treats the window as the complete population rather than a
        sample; the sample deviation would widen every band slightly and
        disagree with standard charting output.
    """
    _require_positive_period(period)
    middle = simple_moving_average(closes, period)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)

    for index in range(period - 1, len(closes)):
        mean = middle[index]
        if mean is None:  # pragma: no cover - guaranteed defined past warm-up
            continue
        window = closes[index - period + 1 : index + 1]
        variance = sum((value - mean) ** 2 for value in window) / period
        deviation = math.sqrt(variance)
        upper[index] = mean + num_std * deviation
        lower[index] = mean - num_std * deviation

    return BollingerResult(upper=upper, middle=middle, lower=lower)


def average_true_range(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Return Wilder's Average True Range.

    Args:
        highs: Session highs, oldest first.
        lows: Session lows.
        closes: Session closes.
        period: Lookback, conventionally 14.

    Returns:
        A list aligned to the inputs.

    Raises:
        ValueError: If the three series have different lengths.

    Notes:
        True Range includes the gap from the previous close, not just the
        intraday range. That is the point of the indicator: a stock that gaps
        down 8% overnight and then trades in a narrow range had a volatile
        session, and high-minus-low alone would call it a calm one.
    """
    _require_positive_period(period)
    if not len(highs) == len(lows) == len(closes):
        msg = "highs, lows and closes must have equal length"
        raise ValueError(msg)

    result: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return result

    true_ranges: list[float] = []
    for index in range(1, len(closes)):
        previous_close = closes[index - 1]
        true_ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - previous_close),
                abs(lows[index] - previous_close),
            )
        )

    average = sum(true_ranges[:period]) / period
    result[period] = average
    for index in range(period, len(true_ranges)):
        average = (average * (period - 1) + true_ranges[index]) / period
        result[index + 1] = average
    return result


def simple_returns(values: Sequence[float], periods: int = 1) -> list[float | None]:
    """Return the fractional change over ``periods`` sessions.

    Args:
        values: Prices, oldest first. Pass *adjusted* closes: a split inside the
            window would otherwise register as a catastrophic loss.
        periods: Sessions to look back. 1 is a daily return.

    Returns:
        A list aligned to ``values``. ``0.05`` means +5%.
    """
    _require_positive_period(periods)
    result: list[float | None] = [None] * len(values)
    for index in range(periods, len(values)):
        base = values[index - periods]
        if base == 0:
            continue
        result[index] = (values[index] - base) / base
    return result


def realised_volatility(
    closes: Sequence[float], *, period: int = 20, annualise: bool = True
) -> list[float | None]:
    """Return the rolling standard deviation of daily returns.

    Args:
        closes: Adjusted closes, oldest first.
        period: Window over which the deviation is measured.
        annualise: Scale by ``sqrt(252)`` so the result reads as an annual
            volatility, which is how volatility is universally quoted.

    Returns:
        A list aligned to ``closes``. ``0.45`` is 45% annualised volatility.

    Notes:
        Uses the *sample* standard deviation, dividing by ``n - 1``. Unlike
        Bollinger bands -- where the window is the population being described --
        here the window is a sample drawn from an unobservable return process,
        and dividing by ``n`` would bias the estimate downward.
    """
    _require_positive_period(period)
    if period < 2:  # noqa: PLR2004 - a deviation needs at least two observations
        msg = "period must be at least 2 to define a deviation"
        raise ValueError(msg)

    returns = simple_returns(closes)
    result: list[float | None] = [None] * len(closes)
    scale = math.sqrt(TRADING_DAYS_PER_YEAR) if annualise else 1.0

    for index in range(len(closes)):
        window = [value for value in returns[index - period + 1 : index + 1] if value is not None]
        if len(window) < period:
            continue
        mean = sum(window) / period
        variance = sum((value - mean) ** 2 for value in window) / (period - 1)
        result[index] = math.sqrt(variance) * scale
    return result


def volume_ratio(volumes: Sequence[float], period: int = 20) -> list[float | None]:
    """Return volume divided by its own moving average.

    Args:
        volumes: Session volumes, oldest first.
        period: Window for the average.

    Returns:
        A list aligned to ``volumes``. ``1.0`` is an average day, ``3.0`` is
        three times normal participation.

    Notes:
        A ratio rather than a raw difference, because volume scale differs by
        orders of magnitude across the tracked universe. Comparing NVIDIA's
        absolute volume to GE Vernova's says nothing; comparing each to its own
        baseline is what makes a spike detectable and cross-sectionally
        comparable.
    """
    _require_positive_period(period)
    averages = simple_moving_average(volumes, period)
    result: list[float | None] = [None] * len(volumes)
    for index, average in enumerate(averages):
        if average is None or average == 0:
            continue
        result[index] = volumes[index] / average
    return result


def relative_strength(
    returns: Sequence[float | None], benchmark_returns: Sequence[float | None]
) -> list[float | None]:
    """Return a series' excess return over a benchmark, session by session.

    Args:
        returns: The instrument's returns.
        benchmark_returns: The benchmark's returns over the same sessions.

    Returns:
        A list aligned to ``returns``; ``None`` wherever either input is.

    Raises:
        ValueError: If the two series have different lengths.

    Notes:
        Separating a company-specific move from a sector-wide one is what makes
        an anomaly explanation meaningful. Micron up 6% on a day the whole
        semiconductor complex rose 5.5% is a sector story; up 6% while the
        sector was flat is a Micron story, and only the second deserves a
        company-specific explanation.
    """
    if len(returns) != len(benchmark_returns):
        msg = "returns and benchmark_returns must have equal length"
        raise ValueError(msg)

    return [
        None if own is None or benchmark is None else own - benchmark
        for own, benchmark in zip(returns, benchmark_returns, strict=True)
    ]


def _rsi_from_averages(average_gain: float, average_loss: float) -> float:
    """Convert Wilder's smoothed averages into a bounded RSI value."""
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative))


def _require_positive_period(period: int) -> None:
    """Reject a non-positive window length."""
    if period < 1:
        msg = "period must be a positive integer"
        raise ValueError(msg)
