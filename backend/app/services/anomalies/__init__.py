"""Anomaly detection: statistics in `detectors`, policy in `anomaly_service`.

Two methods run side by side. The Z-score detector is univariate and can explain
itself; Isolation Forest is multivariate and catches combinations no single
feature reveals. Their verdicts are stored separately so disagreement stays
visible.
"""

from app.services.anomalies.anomaly_service import (
    AnomalyDetectionService,
    AnomalyRunReport,
    TickerAnomalyResult,
)
from app.services.anomalies.detectors import (
    Detection,
    IsolationForestDetector,
    Observation,
    ZScoreDetector,
    robust_z_scores,
)

__all__ = [
    "AnomalyDetectionService",
    "AnomalyRunReport",
    "Detection",
    "IsolationForestDetector",
    "Observation",
    "TickerAnomalyResult",
    "ZScoreDetector",
    "robust_z_scores",
]
