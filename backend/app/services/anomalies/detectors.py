"""Anomaly detectors: robust Z-score and Isolation Forest.

Two methods, deliberately kept separate rather than blended into one score.

The **Z-score** detector is univariate and interpretable. When it fires you can
say exactly why: "volume was 4.2 robust deviations above its median." That
explanation is the product, not a debugging aid -- an anomaly the platform cannot
justify is worthless to an analyst.

**Isolation Forest** is multivariate and catches what no single feature reveals:
a 2% return is unremarkable, and 1.4x volume is unremarkable, but the two
together on a day the sector was flat can be genuinely unusual. It cannot explain
itself in the same way, which is why its verdict is stored alongside rather than
merged with the Z-score's.

Both are stored with the method that produced them, so their disagreement stays
visible instead of being averaged into a number that means neither thing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from app.models.enums import AnomalyType, DetectionMethod, Direction, Severity

#: Scale factor making the median absolute deviation a consistent estimator of
#: the standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826

#: Robust deviations beyond which an observation is reported.
#:
#: Chosen from the false-positive arithmetic rather than by convention. For clean
#: data the scaled MAD approximates sigma, so the Gaussian tail mass applies. The
#: platform scores 14 listings x ~252 sessions x 3 features -- roughly 10,500
#: observations a year -- so the threshold buys:
#:
#:     2.5 sigma -> 1.242% -> ~131 false anomalies a year
#:     3.0 sigma -> 0.270% ->  ~29
#:     3.5 sigma -> 0.047% ->   ~5
#:
#: 3.0 is the compromise: rare enough that the feed stays worth reading, loose
#: enough to catch a genuine sector-wide event. Real returns are fat-tailed, so
#: the observed rate will exceed these figures -- which is the point of also
#: exposing severity, letting a consumer filter to MEDIUM and above.
DEFAULT_Z_THRESHOLD = 3.0

#: Severity ladder, in robust deviations. Chosen so that EXTREME stays rare
#: enough to be worth a notification.
SEVERITY_THRESHOLDS: tuple[tuple[float, Severity], ...] = (
    (8.0, Severity.EXTREME),
    (5.0, Severity.HIGH),
    (3.5, Severity.MEDIUM),
    (0.0, Severity.LOW),
)

#: Sessions of history a detector needs before its verdict means anything. Below
#: this, a "median" is a coin flip and every observation looks extreme.
MIN_HISTORY_SESSIONS = 60

#: Expected outlier fraction for Isolation Forest. Set explicitly rather than
#: left to 'auto': the threshold must not drift as history accumulates, or the
#: same session would be an anomaly one week and not the next.
DEFAULT_CONTAMINATION = 0.02


@dataclass(frozen=True)
class Observation:
    """One session's features, as the detectors consume them."""

    trade_date: date
    daily_return: float | None = None
    volume_ratio: float | None = None
    volatility_20: float | None = None
    relative_strength: float | None = None

    def vector(self, features: Sequence[str]) -> list[float] | None:
        """Return the named features as a dense vector, or ``None`` if incomplete.

        Isolation Forest cannot consume missing values, and imputing them would
        invent the very signal being detected. A session missing a feature is
        skipped instead.
        """
        values: list[float] = []
        for name in features:
            value = getattr(self, name)
            if value is None or not math.isfinite(value):
                return None
            values.append(float(value))
        return values


@dataclass(frozen=True)
class Detection:
    """One detector's verdict on one session."""

    trade_date: date
    anomaly_type: AnomalyType
    method: DetectionMethod
    direction: Direction
    severity: Severity
    score: float
    confidence: float
    observed_value: float | None = None
    expected_value: float | None = None
    deviation: float | None = None
    context: dict[str, Any] = field(default_factory=dict)


def robust_z_scores(values: Sequence[float]) -> tuple[list[float], float, float]:
    """Return robust Z-scores, plus the median and scaled MAD they used.

    Args:
        values: The series to score.

    Returns:
        ``(scores, median, sigma)``. ``sigma`` is zero when the series has no
        dispersion, in which case every score is zero.

    Notes:
        Uses the median and the median absolute deviation, not the mean and
        standard deviation. This is the single most important choice in the
        detector. A 20% single-day move inflates the standard deviation enough to
        pull its own Z-score back under the threshold -- the classic masking
        effect, where the most extreme observation hides itself. The median and
        MAD are unmoved by it, so it stands out at the magnitude it deserves.

        Financial returns are also fat-tailed, so a Gaussian standard deviation
        systematically overstates normal dispersion and under-reports the tails.
    """
    if not values:
        return ([], 0.0, 0.0)

    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    sigma = mad * MAD_TO_SIGMA

    if sigma == 0:
        # A constant series, or one whose middle half is constant. No dispersion
        # means no basis for calling anything an outlier.
        return ([0.0] * len(values), median, 0.0)

    scores = [float((value - median) / sigma) for value in array]
    return (scores, median, sigma)


def severity_for(magnitude: float) -> Severity:
    """Map an absolute deviation onto the severity ladder."""
    for threshold, severity in SEVERITY_THRESHOLDS:
        if magnitude >= threshold:
            return severity
    return Severity.LOW  # pragma: no cover - the ladder ends at 0.0


def confidence_for(magnitude: float, *, threshold: float) -> float:
    """Map a deviation onto a 0-1 confidence.

    Args:
        magnitude: Absolute deviation, in robust sigmas.
        threshold: Deviation at which detection begins.

    Returns:
        Confidence in ``[0, 1]``, saturating rather than clipping.

    Notes:
        A saturating curve rather than a linear ramp with a hard cap. Confidence
        should rise steeply just past the threshold -- where the interesting
        judgement is -- and flatten well before it, because the difference
        between 9 and 12 sigma is not a difference in how sure the platform is.
        Both are certain; only the magnitude differs, and magnitude is reported
        separately.
    """
    if magnitude <= threshold:
        return 0.0
    excess = magnitude - threshold
    return float(1.0 - math.exp(-excess / 2.0))


class ZScoreDetector:
    """Flags univariate outliers in returns, volume and volatility."""

    #: Which feature drives which anomaly type, and whether its sign is
    #: meaningful. A volume spike has no direction -- volume cannot be negative,
    #: and unusually *low* volume is a different phenomenon from a spike.
    _FEATURES: tuple[tuple[str, AnomalyType, bool], ...] = (
        ("daily_return", AnomalyType.RETURN, True),
        ("volume_ratio", AnomalyType.VOLUME, False),
        ("volatility_20", AnomalyType.VOLATILITY, True),
    )

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_Z_THRESHOLD,
        min_history: int = MIN_HISTORY_SESSIONS,
    ) -> None:
        """Configure the detector.

        Args:
            threshold: Robust deviations beyond which a session is reported.
            min_history: Sessions required before any verdict is issued.
        """
        self._threshold = threshold
        self._min_history = min_history

    def detect(
        self, observations: Sequence[Observation], *, since: date | None = None
    ) -> list[Detection]:
        """Score every feature and report the sessions that stand out.

        Args:
            observations: History, oldest first. The whole series is used to
                establish the baseline even when only recent sessions are
                reported.
            since: Report only sessions on or after this date. The baseline is
                still computed from the full history -- narrowing the baseline to
                the reporting window is how a detector ends up comparing a week
                against itself.

        Returns:
            One detection per (session, feature) pair that exceeded the threshold.
        """
        if len(observations) < self._min_history:
            return []

        detections: list[Detection] = []
        for attribute, anomaly_type, signed in self._FEATURES:
            detections.extend(
                self._detect_feature(observations, attribute, anomaly_type, signed, since)
            )
        return detections

    def _detect_feature(
        self,
        observations: Sequence[Observation],
        attribute: str,
        anomaly_type: AnomalyType,
        signed: bool,
        since: date | None,
    ) -> list[Detection]:
        """Score one feature across the history."""
        present = [
            (obs, value)
            for obs in observations
            if (value := getattr(obs, attribute)) is not None and math.isfinite(value)
        ]
        if len(present) < self._min_history:
            return []

        scores, median, sigma = robust_z_scores([value for _, value in present])
        if sigma == 0:
            return []

        detections: list[Detection] = []
        for (obs, value), score in zip(present, scores, strict=True):
            if since is not None and obs.trade_date < since:
                continue
            # Unusually low volume is not a volume spike. Filtering to the upper
            # tail keeps the anomaly type meaning one thing.
            magnitude = abs(score) if signed else max(score, 0.0)
            if magnitude < self._threshold:
                continue

            detections.append(
                Detection(
                    trade_date=obs.trade_date,
                    anomaly_type=anomaly_type,
                    method=DetectionMethod.Z_SCORE,
                    direction=_direction_for(value, signed=signed),
                    severity=severity_for(magnitude),
                    score=round(score, 6),
                    confidence=round(confidence_for(magnitude, threshold=self._threshold), 6),
                    observed_value=value,
                    expected_value=median,
                    deviation=value - median,
                    context={
                        "feature": attribute,
                        "robust_sigma": round(sigma, 8),
                        "threshold": self._threshold,
                        "baseline_sessions": len(present),
                    },
                )
            )
        return detections


class IsolationForestDetector:
    """Flags multivariate outliers using an isolation forest."""

    #: Features forming the vector. Order is fixed and recorded in the stored
    #: context, so a stored score remains interpretable after this list changes.
    FEATURES: tuple[str, ...] = (
        "daily_return",
        "volume_ratio",
        "volatility_20",
        "relative_strength",
    )

    def __init__(
        self,
        *,
        contamination: float = DEFAULT_CONTAMINATION,
        min_history: int = MIN_HISTORY_SESSIONS,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> None:
        """Configure the detector.

        Args:
            contamination: Expected outlier fraction.
            min_history: Sessions required before fitting.
            random_state: Seed. Fixed so that the same history yields the same
                verdicts -- an anomaly that appears and disappears across
                identical runs is not a finding, it is noise.
            n_estimators: Trees in the forest.
        """
        self._contamination = contamination
        self._min_history = min_history
        self._random_state = random_state
        self._n_estimators = n_estimators

    def detect(
        self, observations: Sequence[Observation], *, since: date | None = None
    ) -> list[Detection]:
        """Fit on the history and report the sessions the forest isolates.

        Args:
            observations: History, oldest first.
            since: Report only sessions on or after this date.

        Returns:
            One detection per reported session.

        Notes:
            The model is fit **per ticker, on that ticker's own history**. This is
            the decisive design choice. NVIDIA's ordinary session is a 3% move on
            100M shares; VOO's is 0.4% on 5M. A forest fit across the pooled
            universe would isolate every NVIDIA day as anomalous relative to the
            ETFs, and never notice a genuinely unusual one. Each ticker is
            compared only against its own regime.
        """
        from sklearn.ensemble import IsolationForest  # noqa: PLC0415

        vectors: list[list[float]] = []
        dated: list[Observation] = []
        for obs in observations:
            vector = obs.vector(self.FEATURES)
            if vector is not None:
                vectors.append(vector)
                dated.append(obs)

        if len(vectors) < self._min_history:
            return []

        matrix = np.asarray(vectors, dtype=float)
        forest = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=1,
        )
        forest.fit(matrix)

        # `decision_function` is negative for outliers, with magnitude growing as
        # isolation gets easier. `predict` applies the contamination threshold.
        decisions = forest.decision_function(matrix)
        flags = forest.predict(matrix)

        detections: list[Detection] = []
        for obs, decision, flag in zip(dated, decisions, flags, strict=True):
            if flag != -1:
                continue
            if since is not None and obs.trade_date < since:
                continue

            detections.append(
                Detection(
                    trade_date=obs.trade_date,
                    anomaly_type=AnomalyType.RETURN,
                    method=DetectionMethod.ISOLATION_FOREST,
                    direction=_direction_for(obs.daily_return, signed=True),
                    severity=_severity_from_decision(float(decision)),
                    score=round(float(decision), 6),
                    confidence=round(_confidence_from_decision(float(decision)), 6),
                    observed_value=obs.daily_return,
                    context={
                        "features": list(self.FEATURES),
                        "vector": [round(v, 8) for v in obs.vector(self.FEATURES) or []],
                        "contamination": self._contamination,
                        "baseline_sessions": len(vectors),
                    },
                )
            )
        return detections


def _direction_for(value: float | None, *, signed: bool) -> Direction:
    """Classify the direction of a move."""
    if not signed or value is None:
        return Direction.UP if value is not None and value > 0 else Direction.NEUTRAL
    if value > 0:
        return Direction.UP
    if value < 0:
        return Direction.DOWN
    return Direction.NEUTRAL


def _severity_from_decision(decision: float) -> Severity:
    """Map an isolation-forest decision score onto the severity ladder.

    The score is unbounded below but in practice lies within roughly
    ``[-0.25, 0]`` for outliers, so it is rescaled onto the same sigma-like
    ladder the Z-score detector uses. That keeps a HIGH from one method
    comparable to a HIGH from the other, which is what makes a single severity
    filter across the anomalies feed meaningful.
    """
    return severity_for(abs(decision) * 40.0)


def _confidence_from_decision(decision: float) -> float:
    """Map an isolation-forest decision score onto a 0-1 confidence."""
    return confidence_for(abs(decision) * 40.0, threshold=0.0)
