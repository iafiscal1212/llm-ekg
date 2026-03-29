"""
Security layer for LLM EKG — baseline capture and compromise detection.

Captures the statistical profile of a "healthy" model and detects deviations
that may indicate model compromise, backdoor injection, or silent degradation.

(c) 2026 Carmen Esteban — IAFISCAL & PARTNERS
"""

import json
from pathlib import Path

import numpy as np

from .engine import FEATURE_NAMES, N_FEATURES

# Features with elevated security weight (most likely to shift if compromised)
_SECURITY_WEIGHTS = {
    "hedge_ratio": 1.5,
    "repetition_score": 1.5,
    "confidence_mismatch": 1.5,
    "vocab_diversity": 1.2,
    "specificity_score": 1.2,
    "assertion_density": 1.2,
}


class SecurityReport:
    """Result of comparing current session against a security baseline."""

    def __init__(self, deviations: list, sigma_threshold: float):
        self.deviations = deviations
        self.sigma_threshold = sigma_threshold
        self.n_deviated = sum(1 for d in deviations if d["deviated"])
        if self.n_deviated == 0:
            self.status = "CLEAN"
        elif self.n_deviated <= 3:
            self.status = "WARNING"
        else:
            self.status = "COMPROMISED"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "n_deviated": self.n_deviated,
            "sigma_threshold": self.sigma_threshold,
            "deviations": self.deviations,
        }


class SecurityBaseline:
    """Statistical profile of a known-good model for compromise detection."""

    def __init__(self, feature_stats: list, hurst_stats: list | None = None,
                 state_stats: dict | None = None, n_responses: int = 0,
                 model: str = ""):
        self.feature_stats = feature_stats  # list of dicts, one per feature
        self.hurst_stats = hurst_stats      # list of 16 Hurst values (or None)
        self.state_stats = state_stats or {}
        self.n_responses = n_responses
        self.model = model

    @classmethod
    def from_analyzer(cls, analyzer, model: str = "") -> "SecurityBaseline":
        """Build baseline from an analyzer that has already ingested responses."""
        fh = np.array(analyzer.feature_history)
        n = fh.shape[0]
        if n < 5:
            raise ValueError(f"Need at least 5 responses for baseline, got {n}")

        feature_stats = []
        for i in range(N_FEATURES):
            col = fh[:, i]
            feature_stats.append({
                "name": FEATURE_NAMES[i],
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
                "median": float(np.median(col)),
                "p5": float(np.percentile(col, 5)),
                "p95": float(np.percentile(col, 95)),
            })

        # Hurst exponents from last valid scale analysis
        hurst_stats = None
        for sh in reversed(analyzer.scale_history):
            if sh is not None:
                hurst_stats = sh["hursts"]
                break

        # State engine aggregate metrics
        state_stats = {}
        if analyzer.state_history:
            drifts = [h["drift"] for h in analyzer.state_history]
            anomalies = [h["anomaly_score"] for h in analyzer.state_history]
            state_stats = {
                "drift_mean": float(np.mean(drifts)),
                "drift_std": float(np.std(drifts)),
                "anomaly_mean": float(np.mean(anomalies)),
                "anomaly_std": float(np.std(anomalies)),
            }

        return cls(
            feature_stats=feature_stats,
            hurst_stats=hurst_stats,
            state_stats=state_stats,
            n_responses=n,
            model=model,
        )

    def save(self, path: str):
        """Save baseline to JSON file."""
        data = {
            "version": "1.0",
            "model": self.model,
            "n_responses": self.n_responses,
            "feature_stats": self.feature_stats,
            "hurst_stats": self.hurst_stats,
            "state_stats": self.state_stats,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SecurityBaseline":
        """Load baseline from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_stats=data["feature_stats"],
            hurst_stats=data.get("hurst_stats"),
            state_stats=data.get("state_stats", {}),
            n_responses=data.get("n_responses", 0),
            model=data.get("model", ""),
        )

    def check(self, analyzer, sigma: float = 3.0) -> SecurityReport:
        """Compare current analyzer state against this baseline.

        Args:
            analyzer: LLMAnalyzer with ingested responses.
            sigma: Deviation threshold in standard deviations (default: 3.0).

        Returns:
            SecurityReport with per-feature deviations and overall status.
        """
        fh = np.array(analyzer.feature_history)
        n = fh.shape[0]
        if n == 0:
            return SecurityReport([], sigma)

        deviations = []
        for i in range(N_FEATURES):
            name = FEATURE_NAMES[i]
            bl = self.feature_stats[i]
            current_mean = float(fh[:, i].mean())
            baseline_mean = bl["mean"]
            baseline_std = bl["std"]

            z = abs(current_mean - baseline_mean) / (baseline_std + 1e-10)
            weight = _SECURITY_WEIGHTS.get(name, 1.0)
            weighted_z = z * weight
            deviated = weighted_z > sigma

            deviations.append({
                "name": name,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "current_mean": current_mean,
                "z_score": float(z),
                "weight": weight,
                "weighted_z": float(weighted_z),
                "deviated": deviated,
            })

        # Also check Hurst exponents if available
        hurst_deviations = []
        if self.hurst_stats is not None:
            current_hursts = None
            for sh in reversed(analyzer.scale_history):
                if sh is not None:
                    current_hursts = sh["hursts"]
                    break
            if current_hursts is not None:
                for i, (bh, ch) in enumerate(zip(self.hurst_stats, current_hursts)):
                    delta = abs(ch - bh)
                    # Hurst shift > 0.15 is significant (range is 0-1)
                    deviated = delta > 0.15
                    hurst_deviations.append({
                        "feature": FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}",
                        "baseline_hurst": float(bh),
                        "current_hurst": float(ch),
                        "delta": float(delta),
                        "deviated": deviated,
                    })

        # Count Hurst deviations toward the total
        n_hurst_deviated = sum(1 for h in hurst_deviations if h["deviated"])
        if n_hurst_deviated >= 4:
            # Inject a synthetic deviation to push status
            for d in deviations:
                if not d["deviated"]:
                    d["deviated"] = True
                    d["_hurst_triggered"] = True
                    n_hurst_deviated -= 1
                    if n_hurst_deviated < 1:
                        break

        report = SecurityReport(deviations, sigma)
        report.hurst_deviations = hurst_deviations
        return report
