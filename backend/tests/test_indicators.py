"""Tests for the technical indicator functions.

Indicators are the kind of code that is easy to write plausibly and hard to
write correctly: a wrong smoothing constant or an off-by-one in the warm-up
produces numbers that look like an RSI, move like an RSI, and disagree with
every charting platform a user might check against.

So these tests pin down three things: alignment (a value must sit on the session
that produced it), warm-up (undefined must be ``None``, never zero or a
half-computed value), and arithmetic checked against hand-computed references.
"""

from __future__ import annotations

import math

import pytest

from app.services.features import indicators as ind

#: Wilder's own worked example from *New Concepts in Technical Trading Systems*,
#: the source that defines RSI. Any implementation claiming to compute RSI-14
#: must reproduce these values.
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
    45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
    46.03, 46.41, 46.22, 45.64,
]  # fmt: skip


class TestSimpleMovingAverage:
    """Rolling arithmetic mean."""

    def test_matches_a_hand_computed_mean(self) -> None:
        result = ind.simple_moving_average([1, 2, 3, 4, 5], 3)

        assert result[2] == pytest.approx(2.0)  # (1+2+3)/3
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_warmup_sessions_are_none(self) -> None:
        result = ind.simple_moving_average([1, 2, 3, 4, 5], 3)

        assert result[:2] == [None, None]

    def test_output_is_aligned_to_input(self) -> None:
        values = list(range(50))

        assert len(ind.simple_moving_average(values, 10)) == len(values)

    def test_insufficient_history_yields_all_none(self) -> None:
        assert ind.simple_moving_average([1, 2], 5) == [None, None]

    def test_rolling_update_matches_a_naive_recomputation(self) -> None:
        """The O(n) rolling sum must not drift from a direct mean."""
        values = [float(v) * 1.37 for v in range(1, 200)]
        period = 20

        result = ind.simple_moving_average(values, period)

        for index in range(period - 1, len(values)):
            expected = sum(values[index - period + 1 : index + 1]) / period
            assert result[index] == pytest.approx(expected, rel=1e-9)

    def test_rejects_a_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ind.simple_moving_average([1, 2, 3], 0)


class TestExponentialMovingAverage:
    """Exponentially weighted average, SMA-seeded."""

    def test_seed_is_the_simple_average_of_the_first_window(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = ind.exponential_moving_average(values, 3)

        assert result[2] == pytest.approx(2.0)  # SMA of 1,2,3

    def test_recursion_uses_the_standard_multiplier(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        multiplier = 2 / (3 + 1)

        result = ind.exponential_moving_average(values, 3)

        expected = (4.0 - 2.0) * multiplier + 2.0
        assert result[3] == pytest.approx(expected)

    def test_a_constant_series_stays_at_that_constant(self) -> None:
        result = ind.exponential_moving_average([7.0] * 30, 10)

        assert all(value == pytest.approx(7.0) for value in result[9:])

    def test_reacts_faster_than_the_simple_average(self) -> None:
        """The defining property: recent observations carry more weight."""
        values = [10.0] * 20 + [20.0]

        ema = ind.exponential_moving_average(values, 10)
        sma = ind.simple_moving_average(values, 10)

        assert ema[-1] is not None
        assert sma[-1] is not None
        assert ema[-1] > sma[-1]


class TestRelativeStrengthIndex:
    """Wilder's RSI."""

    def test_matches_the_hand_derived_seed(self) -> None:
        """Check the seed against arithmetic done by hand from the definition.

        Over the 14 changes in ``WILDER_CLOSES``:

            sum(gains)  = 3.34  -> average gain = 3.34 / 14 = 0.2385714
            sum(losses) = 1.40  -> average loss = 1.40 / 14 = 0.1000000
            RS  = 0.2385714 / 0.1 = 2.385714
            RSI = 100 - 100 / (1 + 2.385714) = 70.4641

        Widely circulated tables quote 70.53 for "Wilder's example", but those
        run over a longer price series than the excerpt used here. Deriving the
        expected value from this exact input is what makes the assertion a real
        check rather than a copied constant.
        """
        result = ind.relative_strength_index(WILDER_CLOSES, period=14)

        assert result[14] == pytest.approx(70.4641, abs=0.0001)

    def test_matches_one_hand_derived_smoothing_step(self) -> None:
        """One Wilder step past the seed, again derived from the definition.

        The next change is -0.28, so::

            average gain = (0.2385714 * 13 + 0.00) / 14 = 0.2215306
            average loss = (0.1000000 * 13 + 0.28) / 14 = 0.1128571
            RSI = 100 - 100 / (1 + 0.2215306 / 0.1128571) = 66.2496
        """
        result = ind.relative_strength_index(WILDER_CLOSES, period=14)

        assert result[15] == pytest.approx(66.2496, abs=0.0001)

    def test_warmup_covers_period_sessions(self) -> None:
        """RSI-14 needs 14 *changes*, so 15 prices -- one more than the period."""
        result = ind.relative_strength_index(WILDER_CLOSES, period=14)

        assert result[:14] == [None] * 14
        assert result[14] is not None

    def test_is_bounded_to_zero_and_one_hundred(self) -> None:
        rising = ind.relative_strength_index([float(v) for v in range(1, 60)])
        falling = ind.relative_strength_index([float(v) for v in range(60, 1, -1)])

        assert all(0 <= value <= 100 for value in rising[14:] if value is not None)
        assert all(0 <= value <= 100 for value in falling[14:] if value is not None)

    def test_an_unbroken_advance_reads_one_hundred(self) -> None:
        """No losses means an undefined ratio; RSI is 100 there by definition."""
        result = ind.relative_strength_index([float(v) for v in range(1, 40)])

        assert result[-1] == pytest.approx(100.0)

    def test_an_unbroken_decline_reads_zero(self) -> None:
        result = ind.relative_strength_index([float(v) for v in range(40, 1, -1)])

        assert result[-1] == pytest.approx(0.0)

    def test_uses_wilder_smoothing_not_a_simple_average(self) -> None:
        """The classic wrong implementation: averaging the last N changes.

        A simple average forgets everything outside its window, so a large old
        move vanishes abruptly. Wilder's recursion retains a decaying trace of
        it, which is why the two disagree here.
        """
        closes = [100.0, 110.0] + [100.5 + 0.1 * i for i in range(30)]

        result = ind.relative_strength_index(closes, period=14)

        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        window = changes[-14:]
        naive_gain = sum(max(c, 0) for c in window) / 14
        naive_loss = sum(max(-c, 0) for c in window) / 14
        naive_rsi = 100 - 100 / (1 + naive_gain / naive_loss) if naive_loss else 100.0

        assert result[-1] is not None
        assert result[-1] != pytest.approx(naive_rsi, abs=0.5)

    def test_insufficient_history_yields_all_none(self) -> None:
        assert ind.relative_strength_index([1.0, 2.0, 3.0], period=14) == [None] * 3


class TestMacd:
    """MACD line, signal and histogram."""

    def test_macd_is_the_difference_of_the_two_emas(self) -> None:
        closes = [float(100 + i) for i in range(60)]

        result = ind.macd(closes)
        fast = ind.exponential_moving_average(closes, 12)
        slow = ind.exponential_moving_average(closes, 26)

        index = 40
        assert result.macd[index] == pytest.approx(fast[index] - slow[index])  # type: ignore[operator]

    def test_histogram_is_macd_minus_signal(self) -> None:
        closes = [float(100 + (i % 7) * 2) for i in range(80)]

        result = ind.macd(closes)

        for m, s, h in zip(result.macd, result.signal, result.histogram, strict=True):
            if m is None or s is None:
                assert h is None
            else:
                assert h == pytest.approx(m - s)

    def test_signal_warmup_starts_after_the_macd_line(self) -> None:
        """The signal EMA runs over MACD values, so it warms up strictly later."""
        closes = [float(100 + i) for i in range(80)]

        result = ind.macd(closes)

        first_macd = next(i for i, v in enumerate(result.macd) if v is not None)
        first_signal = next(i for i, v in enumerate(result.signal) if v is not None)
        assert first_signal > first_macd
        assert first_signal == first_macd + 8  # signal period 9, minus one

    def test_rising_series_gives_a_positive_macd(self) -> None:
        closes = [float(100 + i * 2) for i in range(80)]

        result = ind.macd(closes)

        assert result.macd[-1] is not None
        assert result.macd[-1] > 0

    def test_rejects_an_inverted_period_pair(self) -> None:
        with pytest.raises(ValueError, match="shorter"):
            ind.macd([1.0] * 50, fast_period=26, slow_period=12)


class TestBollingerBands:
    """Bands around a moving average."""

    def test_bands_sit_symmetrically_around_the_middle(self) -> None:
        closes = [float(100 + (i % 5)) for i in range(40)]

        bands = ind.bollinger_bands(closes, period=20)

        index = 30
        upper, middle, lower = bands.upper[index], bands.middle[index], bands.lower[index]
        assert upper is not None and middle is not None and lower is not None
        assert upper - middle == pytest.approx(middle - lower)

    def test_uses_the_population_deviation(self) -> None:
        """Bollinger treats the window as a population, dividing by n not n-1."""
        closes = [float(v) for v in range(1, 21)]

        bands = ind.bollinger_bands(closes, period=20, num_std=1)

        mean = sum(closes) / 20
        population_std = math.sqrt(sum((c - mean) ** 2 for c in closes) / 20)
        assert bands.upper[19] == pytest.approx(mean + population_std)

    def test_a_flat_series_collapses_the_bands(self) -> None:
        bands = ind.bollinger_bands([50.0] * 30, period=20)

        assert bands.upper[-1] == pytest.approx(50.0)
        assert bands.lower[-1] == pytest.approx(50.0)

    def test_middle_band_equals_the_simple_moving_average(self) -> None:
        closes = [float(100 + (i % 11)) for i in range(50)]

        bands = ind.bollinger_bands(closes, period=20)
        sma = ind.simple_moving_average(closes, 20)

        assert bands.middle == sma


class TestAverageTrueRange:
    """Wilder's ATR."""

    def test_true_range_includes_the_overnight_gap(self) -> None:
        """The point of ATR: a gap is volatility, even with a narrow session.

        Both series have identical intraday ranges; only the second gaps. ATR
        must be larger for it -- high-minus-low alone would call them equal.
        """
        highs_flat = [101.0] * 20
        lows_flat = [99.0] * 20
        closes_flat = [100.0] * 20

        highs_gap = [101.0] * 10 + [121.0] * 10
        lows_gap = [99.0] * 10 + [119.0] * 10
        closes_gap = [100.0] * 10 + [120.0] * 10

        flat = ind.average_true_range(highs_flat, lows_flat, closes_flat, period=5)
        gapped = ind.average_true_range(highs_gap, lows_gap, closes_gap, period=5)

        assert flat[-1] is not None
        assert gapped[-1] is not None
        assert gapped[-1] > flat[-1]

    def test_constant_range_gives_that_range(self) -> None:
        highs = [102.0] * 30
        lows = [100.0] * 30
        closes = [101.0] * 30

        result = ind.average_true_range(highs, lows, closes, period=14)

        assert result[-1] == pytest.approx(2.0)

    def test_is_never_negative(self) -> None:
        highs = [100.0 + (i % 5) for i in range(60)]
        lows = [95.0 - (i % 3) for i in range(60)]
        closes = [98.0 + (i % 4) for i in range(60)]

        result = ind.average_true_range(highs, lows, closes)

        assert all(value >= 0 for value in result if value is not None)

    def test_mismatched_series_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            ind.average_true_range([1.0, 2.0], [1.0], [1.0, 2.0])


class TestReturns:
    """Fractional price changes."""

    def test_daily_return_is_the_fractional_change(self) -> None:
        result = ind.simple_returns([100.0, 110.0, 99.0])

        assert result[1] == pytest.approx(0.10)
        assert result[2] == pytest.approx(-0.10)

    def test_multi_session_lookback(self) -> None:
        result = ind.simple_returns([100.0, 105.0, 110.0, 120.0], periods=3)

        assert result[3] == pytest.approx(0.20)
        assert result[:3] == [None, None, None]

    def test_a_zero_base_price_yields_none_not_a_division_error(self) -> None:
        result = ind.simple_returns([0.0, 10.0])

        assert result[1] is None


class TestRealisedVolatility:
    """Rolling standard deviation of returns."""

    def test_a_flat_series_has_zero_volatility(self) -> None:
        result = ind.realised_volatility([100.0] * 40, period=20)

        assert result[-1] == pytest.approx(0.0)

    def test_annualisation_scales_by_root_252(self) -> None:
        closes = [100.0 * (1.01 if i % 2 else 0.99) for i in range(60)]

        annualised = ind.realised_volatility(closes, period=20)
        daily = ind.realised_volatility(closes, period=20, annualise=False)

        assert annualised[-1] == pytest.approx(daily[-1] * math.sqrt(252))  # type: ignore[operator]

    def test_a_more_volatile_series_reads_higher(self) -> None:
        calm = [100.0 + (i % 2) * 0.1 for i in range(60)]
        wild = [100.0 + (i % 2) * 10.0 for i in range(60)]

        assert ind.realised_volatility(calm)[-1] < ind.realised_volatility(wild)[-1]  # type: ignore[operator]

    def test_uses_the_sample_deviation(self) -> None:
        """A sample of an unobservable process, so n-1 -- unlike Bollinger."""
        closes = [100.0, 101.0, 103.0, 102.0, 105.0, 104.0]

        result = ind.realised_volatility(closes, period=5, annualise=False)

        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1] for i in range(len(closes) - 5, len(closes))
        ]
        mean = sum(returns) / 5
        sample_std = math.sqrt(sum((r - mean) ** 2 for r in returns) / 4)
        assert result[-1] == pytest.approx(sample_std)

    def test_a_period_below_two_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ind.realised_volatility([1.0, 2.0, 3.0], period=1)


class TestVolumeRatio:
    """Volume against its own baseline."""

    def test_average_volume_reads_one(self) -> None:
        result = ind.volume_ratio([1000.0] * 30, period=20)

        assert result[-1] == pytest.approx(1.0)

    def test_a_spike_reads_above_one(self) -> None:
        volumes = [1000.0] * 25 + [5000.0]

        result = ind.volume_ratio(volumes, period=20)

        assert result[-1] is not None
        assert result[-1] > 4.0

    def test_is_comparable_across_wildly_different_scales(self) -> None:
        """The reason it is a ratio: absolute volumes are not comparable."""
        small = [1_000.0] * 25 + [3_000.0]
        large = [500_000_000.0] * 25 + [1_500_000_000.0]

        assert ind.volume_ratio(small)[-1] == pytest.approx(ind.volume_ratio(large)[-1])  # type: ignore[arg-type]

    def test_zero_baseline_yields_none(self) -> None:
        result = ind.volume_ratio([0.0] * 25, period=20)

        assert result[-1] is None


class TestRelativeStrength:
    """Excess return over a benchmark."""

    def test_excess_return_is_the_difference(self) -> None:
        result = ind.relative_strength([0.06, 0.02], [0.055, -0.01])

        assert result[0] == pytest.approx(0.005)
        assert result[1] == pytest.approx(0.03)

    def test_a_sector_wide_move_leaves_little_excess(self) -> None:
        """Up 6% while the sector rose 5.5% is a sector story, not a stock one."""
        result = ind.relative_strength([0.06], [0.055])

        assert result[0] is not None
        assert abs(result[0]) < 0.01

    def test_missing_benchmark_data_propagates_as_none(self) -> None:
        result = ind.relative_strength([0.01, 0.02], [None, 0.005])

        assert result[0] is None
        assert result[1] == pytest.approx(0.015)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            ind.relative_strength([0.01], [0.01, 0.02])


class TestAlignmentInvariant:
    """The contract every function in the module shares."""

    @pytest.mark.parametrize("length", [0, 1, 5, 37, 300])
    def test_every_indicator_returns_a_list_matching_its_input(self, length: int) -> None:
        closes = [100.0 + i * 0.5 for i in range(length)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        volumes = [1_000_000.0] * length

        assert len(ind.simple_moving_average(closes, 20)) == length
        assert len(ind.exponential_moving_average(closes, 12)) == length
        assert len(ind.relative_strength_index(closes)) == length
        assert len(ind.average_true_range(highs, lows, closes)) == length
        assert len(ind.simple_returns(closes)) == length
        assert len(ind.realised_volatility(closes)) == length
        assert len(ind.volume_ratio(volumes)) == length

        macd_result = ind.macd(closes)
        assert len(macd_result.macd) == length
        assert len(macd_result.signal) == length
        assert len(macd_result.histogram) == length

        bands = ind.bollinger_bands(closes)
        assert len(bands.upper) == len(bands.middle) == len(bands.lower) == length
