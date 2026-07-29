"""Tests for the anomaly detectors.

The properties that matter are not "does it return a number" but "does it fire
on the thing an analyst would call unusual, and stay quiet otherwise". A detector
that flags everything is as useless as one that flags nothing, so most of these
tests pin down the *absence* of detections on ordinary data.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from app.models.enums import AnomalyType, DetectionMethod, Direction, Severity
from app.services.anomalies.detectors import (
    DEFAULT_Z_THRESHOLD,
    IsolationForestDetector,
    Observation,
    ZScoreDetector,
    confidence_for,
    robust_z_scores,
    severity_for,
)

START = date(2024, 1, 1)


def _observations(
    returns: list[float],
    *,
    volume_ratios: list[float] | None = None,
    volatilities: list[float] | None = None,
) -> list[Observation]:
    """Build a history from parallel feature series."""
    count = len(returns)
    volumes = volume_ratios or [1.0] * count
    vols = volatilities or [0.30] * count
    return [
        Observation(
            trade_date=START + timedelta(days=index),
            daily_return=returns[index],
            volume_ratio=volumes[index],
            volatility_20=vols[index],
            relative_strength=returns[index] * 0.5,
        )
        for index in range(count)
    ]


def _calm_returns(count: int, *, seed: int = 7) -> list[float]:
    """A quiet series: small moves, no outliers."""
    rng = random.Random(seed)
    return [rng.gauss(0.0005, 0.008) for _ in range(count)]


class TestRobustZScores:
    """The statistic the whole Z-score detector rests on."""

    def test_a_constant_series_has_no_dispersion(self) -> None:
        scores, median, sigma = robust_z_scores([5.0] * 50)

        assert sigma == 0
        assert median == 5.0
        assert all(score == 0 for score in scores)

    def test_scores_are_centred_on_the_median(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        scores, median, _ = robust_z_scores(values)

        assert median == 3.0
        assert scores[2] == pytest.approx(0.0)
        assert scores[0] < 0 < scores[4]

    def test_an_outlier_does_not_mask_itself(self) -> None:
        """The reason for median/MAD over mean/std, demonstrated directly.

        A single extreme value inflates the standard deviation enough to pull
        its own Z-score back under the threshold. The MAD is unmoved by it.
        """
        values = [*_calm_returns(200), 0.35]

        robust, _, _ = robust_z_scores(values)

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        classical = (values[-1] - mean) / math.sqrt(variance)

        assert abs(robust[-1]) > 20
        assert abs(classical) < 15
        assert abs(robust[-1]) > abs(classical)

    def test_empty_input_is_handled(self) -> None:
        assert robust_z_scores([]) == ([], 0.0, 0.0)


class TestSeverityAndConfidence:
    """The ladder that makes two detectors comparable."""

    @pytest.mark.parametrize(
        ("magnitude", "expected"),
        [
            (2.6, Severity.LOW),
            (4.0, Severity.MEDIUM),
            (6.0, Severity.HIGH),
            (12.0, Severity.EXTREME),
        ],
    )
    def test_severity_ladder(self, magnitude: float, expected: Severity) -> None:
        assert severity_for(magnitude) is expected

    def test_confidence_is_zero_at_the_threshold(self) -> None:
        assert confidence_for(2.5, threshold=2.5) == 0.0

    def test_confidence_rises_then_saturates(self) -> None:
        """Steep where the judgement is, flat where it no longer matters."""
        near = confidence_for(3.5, threshold=2.5)
        far = confidence_for(9.0, threshold=2.5)
        further = confidence_for(15.0, threshold=2.5)

        assert 0 < near < far < further < 1
        # Past ~9 sigma the curve is flat: 9 and 15 are both "certain", and the
        # difference between them is magnitude, which is reported separately.
        assert far > 0.95
        assert further - far < 0.05

    def test_confidence_never_exceeds_one(self) -> None:
        assert confidence_for(1000.0, threshold=2.5) <= 1.0


class TestZScoreDetector:
    """Univariate detection over returns, volume and volatility."""

    def test_a_calm_series_stays_within_the_expected_tail_mass(self) -> None:
        """A detector that fires on ordinary data is noise, not a feature.

        Zero detections would be the wrong assertion: at 3 sigma the Gaussian
        tail is 0.27%, so a few hits across 300 sessions x 3 features is the
        statistics working correctly. What must hold is that the rate stays
        near that bound rather than an order of magnitude above it.
        """
        detector = ZScoreDetector()

        detections = detector.detect(_observations(_calm_returns(300)))

        observations_scored = 300 * 3  # three features
        assert len(detections) <= observations_scored * 0.01

    def test_a_large_move_is_detected(self) -> None:
        returns = [*_calm_returns(250), 0.22]
        detector = ZScoreDetector()

        detections = detector.detect(_observations(returns))

        returns_found = [d for d in detections if d.anomaly_type is AnomalyType.RETURN]
        planted = _observations(returns)[-1].trade_date
        biggest = max(returns_found, key=lambda d: abs(d.score))
        assert biggest.trade_date == planted
        assert biggest.method is DetectionMethod.Z_SCORE
        assert biggest.direction is Direction.UP
        assert biggest.severity in {Severity.HIGH, Severity.EXTREME}

    def test_direction_reflects_the_sign_of_the_move(self) -> None:
        detector = ZScoreDetector()

        crash = detector.detect(_observations([*_calm_returns(250), -0.22]))

        found = next(d for d in crash if d.anomaly_type is AnomalyType.RETURN)
        assert found.direction is Direction.DOWN

    def test_a_volume_spike_is_detected(self) -> None:
        count = 250
        volumes = [1.0 + (index % 5) * 0.05 for index in range(count)] + [7.0]
        detector = ZScoreDetector()

        detections = detector.detect(_observations(_calm_returns(count + 1), volume_ratios=volumes))

        volume_found = [d for d in detections if d.anomaly_type is AnomalyType.VOLUME]
        assert len(volume_found) == 1
        assert volume_found[0].observed_value == pytest.approx(7.0)

    def test_unusually_low_volume_is_not_a_volume_spike(self) -> None:
        """A quiet day is a different phenomenon and must not share the label."""
        count = 250
        volumes = [1.0 + (index % 5) * 0.05 for index in range(count)] + [0.01]
        detector = ZScoreDetector()

        detections = detector.detect(_observations(_calm_returns(count + 1), volume_ratios=volumes))

        assert [d for d in detections if d.anomaly_type is AnomalyType.VOLUME] == []

    def test_short_history_produces_nothing(self) -> None:
        """With 20 sessions a median is a coin flip and everything looks extreme."""
        detector = ZScoreDetector()

        assert detector.detect(_observations(_calm_returns(20))) == []

    def test_the_baseline_uses_full_history_not_just_the_reported_window(self) -> None:
        """Narrowing the baseline to the window compares a week against itself."""
        returns = [*_calm_returns(250), 0.22]
        observations = _observations(returns)
        detector = ZScoreDetector()

        recent_only = observations[-1].trade_date
        detections = detector.detect(observations, since=recent_only)

        found = [d for d in detections if d.anomaly_type is AnomalyType.RETURN]
        assert len(found) == 1
        # The baseline still saw 250 calm sessions, so the move reads as extreme.
        assert abs(found[0].score) > 10

    def test_since_filters_what_is_reported(self) -> None:
        returns = [*_calm_returns(120), 0.22, *_calm_returns(120, seed=9)]
        observations = _observations(returns)
        detector = ZScoreDetector()

        after_the_spike = observations[130].trade_date
        detections = detector.detect(observations, since=after_the_spike)

        assert all(d.trade_date >= after_the_spike for d in detections)

    def test_context_records_how_the_verdict_was_reached(self) -> None:
        """An anomaly the platform cannot justify is worthless to an analyst."""
        detections = ZScoreDetector().detect(_observations([*_calm_returns(250), 0.22]))

        found = next(d for d in detections if d.anomaly_type is AnomalyType.RETURN)
        assert found.context["feature"] == "daily_return"
        assert found.context["robust_sigma"] > 0
        assert found.context["threshold"] == DEFAULT_Z_THRESHOLD
        assert found.context["baseline_sessions"] == 251
        assert found.expected_value is not None
        assert found.deviation is not None

    def test_a_flat_feature_cannot_produce_detections(self) -> None:
        """No dispersion means no basis for calling anything an outlier."""
        detector = ZScoreDetector()
        observations = _observations([0.0] * 300)

        assert detector.detect(observations) == []


class TestIsolationForestDetector:
    """Multivariate detection."""

    def test_a_calm_series_produces_few_detections(self) -> None:
        """Contamination bounds the false-positive rate by construction."""
        detector = IsolationForestDetector(contamination=0.02)

        detections = detector.detect(_observations(_calm_returns(300)))

        assert len(detections) <= 300 * 0.03

    def test_a_combined_outlier_is_detected(self) -> None:
        """Neither feature is extreme alone; together they are unusual."""
        count = 250
        returns = [*_calm_returns(count), 0.05]
        volumes = [1.0 + (index % 4) * 0.05 for index in range(count)] + [4.5]
        observations = _observations(returns, volume_ratios=volumes)

        detections = IsolationForestDetector().detect(observations)

        flagged = {detection.trade_date for detection in detections}
        assert observations[-1].trade_date in flagged

    def test_results_are_deterministic(self) -> None:
        """An anomaly that comes and goes across identical runs is noise."""
        observations = _observations([*_calm_returns(250), 0.18])

        first = IsolationForestDetector().detect(observations)
        second = IsolationForestDetector().detect(observations)

        assert [(d.trade_date, d.score) for d in first] == [(d.trade_date, d.score) for d in second]

    def test_sessions_missing_a_feature_are_skipped_not_imputed(self) -> None:
        """Imputing would invent the very signal being detected."""
        observations = _observations(_calm_returns(200))
        observations[50] = Observation(trade_date=observations[50].trade_date, daily_return=0.01)

        detections = IsolationForestDetector().detect(observations)

        assert observations[50].trade_date not in {d.trade_date for d in detections}

    def test_short_history_produces_nothing(self) -> None:
        assert IsolationForestDetector().detect(_observations(_calm_returns(30))) == []

    def test_context_records_the_feature_vector(self) -> None:
        observations = _observations([*_calm_returns(250), 0.20])

        detections = IsolationForestDetector().detect(observations)

        assert detections
        context = detections[0].context
        assert context["features"] == list(IsolationForestDetector.FEATURES)
        assert len(context["vector"]) == len(IsolationForestDetector.FEATURES)
        assert context["contamination"] == 0.02

    def test_method_is_recorded_so_verdicts_stay_distinguishable(self) -> None:
        detections = IsolationForestDetector().detect(_observations([*_calm_returns(250), 0.20]))

        assert all(d.method is DetectionMethod.ISOLATION_FOREST for d in detections)


class TestDetectorsTogether:
    """The two methods are stored side by side, not merged."""

    def test_both_can_fire_on_the_same_session_independently(self) -> None:
        observations = _observations([*_calm_returns(250), 0.25])

        z_detections = ZScoreDetector().detect(observations)
        forest_detections = IsolationForestDetector().detect(observations)

        last = observations[-1].trade_date
        assert last in {d.trade_date for d in z_detections}
        assert last in {d.trade_date for d in forest_detections}

    def test_their_scores_are_not_comparable_but_severities_are(self) -> None:
        """Raw scores mean different things; the severity ladder is shared."""
        observations = _observations([*_calm_returns(250), 0.25])

        z_score = next(
            d
            for d in ZScoreDetector().detect(observations)
            if d.trade_date == observations[-1].trade_date
        )
        forest = next(
            d
            for d in IsolationForestDetector().detect(observations)
            if d.trade_date == observations[-1].trade_date
        )

        assert z_score.score != forest.score
        assert isinstance(z_score.severity, Severity)
        assert isinstance(forest.severity, Severity)
        assert 0 <= z_score.confidence <= 1
        assert 0 <= forest.confidence <= 1
