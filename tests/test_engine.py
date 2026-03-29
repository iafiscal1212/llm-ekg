"""Tests for LLM EKG core engine: feature extraction, state engine, analyzer.

Covers the critical mathematical components with NO hardcoded features —
all tests use real text inputs and verify structural invariants.
"""

import numpy as np
import pytest

from llm_ekg.engine import (
    LLMAnalyzer,
    LLMFeatureExtractor,
    FEATURE_NAMES,
    N_FEATURES,
    _StateEngine,
    _FreqAnalyzer,
    _hl,
)


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    """LLMFeatureExtractor.extract() — 16 numerical features from text."""

    @pytest.fixture
    def extractor(self):
        return LLMFeatureExtractor()

    def test_returns_16_features(self, extractor):
        v = extractor.extract("Hello world, this is a test.", 0.5)
        assert v.shape == (N_FEATURES,)
        assert v.dtype == np.float64

    def test_empty_string_returns_zeros(self, extractor):
        v = extractor.extract("", 0.0)
        assert np.all(v == 0.0)

    def test_whitespace_only_returns_zeros(self, extractor):
        v = extractor.extract("   \n\t  ", 0.0)
        assert np.all(v == 0.0)

    def test_length_chars(self, extractor):
        text = "Hello world"
        v = extractor.extract(text, 0.0)
        assert v[0] == len(text)

    def test_length_words(self, extractor):
        v = extractor.extract("one two three four five", 0.0)
        assert v[1] == 5

    def test_vocab_diversity_range(self, extractor):
        v = extractor.extract("the the the the the", 0.0)
        assert 0.0 <= v[2] <= 1.0

    def test_high_vocab_diversity(self, extractor):
        v = extractor.extract("alpha bravo charlie delta echo", 0.0)
        assert v[2] > 0.8  # all unique words

    def test_response_time_passthrough(self, extractor):
        v = extractor.extract("test", 3.14)
        assert v[11] == pytest.approx(3.14)

    def test_hedge_ratio_with_hedges(self, extractor):
        hedgy = ("Perhaps the data might suggest some correlation, "
                 "although it could possibly be coincidental. "
                 "Generally speaking, the trends seem somewhat positive.")
        v = extractor.extract(hedgy, 0.0)
        assert v[7] > 0.05  # hedge_ratio should be non-trivial

    def test_hedge_ratio_without_hedges(self, extractor):
        direct = ("The revenue increased by 15% year over year. "
                  "Operating margins improved to 18.5% from 16.1%.")
        v = extractor.extract(direct, 0.0)
        assert v[7] < 0.05  # very few hedges

    def test_list_ratio_with_lists(self, extractor):
        text = "Key points:\n- First item\n- Second item\n- Third item"
        v = extractor.extract(text, 0.0)
        assert v[8] > 0.5  # list_ratio should be high

    def test_code_ratio_with_code(self, extractor):
        text = "Here is code:\n```python\nprint('hello')\n```\nEnd."
        v = extractor.extract(text, 0.0)
        assert v[9] > 0.0  # code_ratio should be non-zero

    def test_specificity_with_numbers(self, extractor):
        specific = ("Revenue: $42.7M (+15.3%). Operating margin: 18.5%. "
                    "Q4 2025 results as of 12/31/2025.")
        v = extractor.extract(specific, 0.0)
        assert v[12] > 0.5  # specificity_score should be high

    def test_assertion_density_with_assertions(self, extractor):
        assertive = ("The system is always reliable. Every module must "
                     "precisely handle edge cases. It never fails.")
        v = extractor.extract(assertive, 0.0)
        assert v[14] > 0.5  # assertion_density should be high

    def test_assertion_density_with_conditionals(self, extractor):
        conditional = ("If the system might fail, it could possibly "
                       "require attention, unless conditions change.")
        v = extractor.extract(conditional, 0.0)
        assert v[14] < 0.5  # assertion_density should be low

    def test_all_features_finite(self, extractor):
        texts = [
            "Short text.",
            "A" * 10000,
            "perhaps maybe might could possibly",
            "1. Item\n2. Item\n3. Item",
            "The revenue was $42.7M in Q4 2025 at 15.3% growth.",
        ]
        for text in texts:
            v = extractor.extract(text, 1.0)
            assert np.all(np.isfinite(v)), f"Non-finite features for: {text[:30]}"

    def test_repetition_score_builds_over_calls(self, extractor):
        text = "The quick brown fox jumps over the lazy dog."
        extractor.extract(text, 0.0)  # first call: no prev
        v2 = extractor.extract(text, 0.0)  # second call: same text
        assert v2[10] > 0.0  # repetition_score > 0 for identical text


# ---------------------------------------------------------------------------
# State Engine
# ---------------------------------------------------------------------------

class TestStateEngine:
    """_StateEngine — behavioral dynamics with Hebbian learning."""

    @pytest.fixture
    def engine(self):
        return _StateEngine(input_dim=N_FEATURES, hidden_dim=48, seed=42)

    def test_initial_h_is_zeros(self, engine):
        assert np.all(engine.h == 0.0)

    def test_step_returns_expected_keys(self, engine):
        obs = np.random.randn(N_FEATURES)
        result = engine.step(obs)
        expected_keys = {
            "metrics", "m0_memory", "m1_variability", "m2_persistence",
            "m3_complexity", "drift", "state_change", "self_ratio",
            "energy_rate", "anomaly_score", "h_norm", "step",
        }
        assert set(result.keys()) == expected_keys

    def test_h_norm_bounded(self, engine):
        """h_norm should never exceed 10.0 (hard clamp in step)."""
        for _ in range(100):
            obs = np.random.randn(N_FEATURES) * 10  # large inputs
            result = engine.step(obs)
        assert result["h_norm"] <= 10.0 + 1e-10

    def test_anomaly_score_range(self, engine):
        """anomaly_score should be in [0, 1] after warmup."""
        for _ in range(30):
            obs = np.random.randn(N_FEATURES)
            result = engine.step(obs)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_anomaly_zero_before_warmup(self, engine):
        """anomaly_score is 0 during first 20 steps (warmup)."""
        for i in range(19):
            obs = np.random.randn(N_FEATURES)
            result = engine.step(obs)
            assert result["anomaly_score"] == 0.0, f"Non-zero at step {i}"

    def test_step_counter_increments(self, engine):
        for i in range(5):
            result = engine.step(np.zeros(N_FEATURES))
            assert result["step"] == i + 1

    def test_drift_first_step_is_zero(self, engine):
        result = engine.step(np.zeros(N_FEATURES))
        assert result["drift"] == 0.0

    def test_metrics_has_four_components(self, engine):
        result = engine.step(np.random.randn(N_FEATURES))
        assert len(result["metrics"]) == 4

    def test_deterministic_with_same_seed(self):
        """Same seed + same inputs = same outputs."""
        e1 = _StateEngine(input_dim=N_FEATURES, hidden_dim=48, seed=42)
        e2 = _StateEngine(input_dim=N_FEATURES, hidden_dim=48, seed=42)
        np.random.seed(99)
        obs = np.random.randn(N_FEATURES)
        # Note: step() uses np.random internally, so we set seed before each
        np.random.seed(123)
        r1 = e1.step(obs)
        np.random.seed(123)
        r2 = e2.step(obs)
        assert r1["h_norm"] == pytest.approx(r2["h_norm"])
        assert r1["drift"] == pytest.approx(r2["drift"])

    def test_no_nan_after_many_steps(self, engine):
        """Engine should never produce NaN even after many steps."""
        for _ in range(200):
            obs = np.random.randn(N_FEATURES) * 5
            result = engine.step(obs)
        assert np.isfinite(result["h_norm"])
        assert np.isfinite(result["drift"])
        assert np.isfinite(result["anomaly_score"])


# ---------------------------------------------------------------------------
# Frequency Analyzer
# ---------------------------------------------------------------------------

class TestFreqAnalyzer:
    """_FreqAnalyzer — multi-scale frequency analysis with Hurst exponent."""

    @pytest.fixture
    def freq(self):
        return _FreqAnalyzer(n_gen=6, fft_window=32)

    def test_insufficient_data_returns_default(self, freq):
        series = np.random.randn(10)  # < fft_window
        result = freq.analyze(series)
        assert np.all(result["band_energies"] == 0.0)
        assert result["hurst"] == 0.5

    def test_hurst_in_valid_range(self, freq):
        series = np.random.randn(64)
        result = freq.analyze(series)
        assert 0.01 <= result["hurst"] <= 0.99

    def test_band_energies_shape(self, freq):
        series = np.random.randn(64)
        result = freq.analyze(series)
        assert result["band_energies"].shape == (6,)

    def test_analyze_all_shape(self, freq):
        fh = np.random.randn(64, N_FEATURES)
        result = freq.analyze_all(fh)
        assert len(result["per_feature"]) == N_FEATURES
        assert len(result["hursts"]) == N_FEATURES
        assert 0.01 <= result["mean_hurst"] <= 0.99

    def test_hurst_labels_valid(self, freq):
        fh = np.random.randn(64, N_FEATURES)
        result = freq.analyze_all(fh)
        valid = {"Trending", "Persistent", "Random", "Mean-Revert", "Anti-Persist"}
        for label in result["hurst_labels"]:
            assert label in valid


class TestHurstLabel:
    """_hl() — Hurst exponent to human-readable label."""

    def test_trending(self):
        assert _hl(0.7) == "Trending"

    def test_persistent(self):
        assert _hl(0.6) == "Persistent"

    def test_random(self):
        assert _hl(0.5) == "Random"

    def test_mean_revert(self):
        assert _hl(0.4) == "Mean-Revert"

    def test_anti_persist(self):
        assert _hl(0.3) == "Anti-Persist"


# ---------------------------------------------------------------------------
# LLMAnalyzer (integration)
# ---------------------------------------------------------------------------

class TestLLMAnalyzer:
    """LLMAnalyzer — full pipeline: extract → engine → frequency."""

    @pytest.fixture
    def analyzer(self):
        return LLMAnalyzer()

    def test_ingest_returns_expected_keys(self, analyzer):
        result = analyzer.ingest("Hello world.", timestamp=1.0)
        assert "step" in result
        assert "features" in result
        assert "state" in result
        assert "multiscale" in result

    def test_ingest_features_are_dict(self, analyzer):
        result = analyzer.ingest("Test response.", timestamp=1.0)
        assert isinstance(result["features"], dict)
        assert set(result["features"].keys()) == set(FEATURE_NAMES)

    def test_history_grows(self, analyzer):
        for i in range(5):
            analyzer.ingest(f"Response number {i}.", timestamp=float(i))
        assert len(analyzer.feature_history) == 5
        assert len(analyzer.state_history) == 5
        assert len(analyzer.timestamps) == 5

    def test_multiscale_none_before_window(self, analyzer):
        result = analyzer.ingest("Short session.", timestamp=1.0)
        assert result["multiscale"] is None

    def test_multiscale_available_after_window(self, analyzer):
        for i in range(35):
            result = analyzer.ingest(
                f"Response {i} with varied content about topic {i % 5}.",
                timestamp=float(i),
            )
        assert result["multiscale"] is not None
        assert "mean_hurst" in result["multiscale"]

    def test_ingest_batch(self, analyzer):
        responses = [
            {"response": f"Batch item {i}.", "timestamp": float(i)}
            for i in range(5)
        ]
        results = analyzer.ingest_batch(responses)
        assert len(results) == 5
        assert all("step" in r for r in results)

    def test_get_summary_no_data(self, analyzer):
        summary = analyzer.get_summary()
        assert summary["n_responses"] == 0
        assert summary["verdict"] == "NO DATA"

    def test_get_summary_after_ingest(self, analyzer):
        for i in range(10):
            analyzer.ingest(
                f"Response {i}: The quarterly revenue was ${i * 10}M.",
                timestamp=float(i),
                response_time_s=0.5,
            )
        summary = analyzer.get_summary()
        assert summary["n_responses"] == 10
        assert 0 <= summary["global_score_100"] <= 100
        assert summary["verdict"] in ("HEALTHY", "DEGRADED", "CRITICAL")
        assert 0.0 <= summary["hallucination_risk"] <= 1.0

    def test_get_summary_keys(self, analyzer):
        analyzer.ingest("Test.", timestamp=1.0)
        summary = analyzer.get_summary()
        expected = {
            "n_responses", "global_score_100", "global_anomaly_mean",
            "recent_anomaly_mean", "trend", "verdict", "drift_mean",
            "m0_mean", "m3_mean", "multiscale", "specificity_mean",
            "confidence_mismatch_mean", "assertion_density_mean",
            "self_consistency_mean", "hallucination_risk",
            "recent_hallucination_risk",
        }
        assert set(summary.keys()) == expected

    def test_healthy_responses_score_high(self, analyzer):
        """Consistent, specific responses should score high."""
        for i in range(20):
            analyzer.ingest(
                f"Revenue was ${42 + i}M in Q{(i % 4) + 1} 2025. "
                f"Operating margin: {18 + i * 0.1:.1f}%. "
                f"Year-over-year growth: {15 + i * 0.2:.1f}%.",
                timestamp=float(i),
                response_time_s=0.5,
            )
        summary = analyzer.get_summary()
        assert summary["global_score_100"] >= 70

    def test_feature_history_is_numpy(self, analyzer):
        analyzer.ingest("Test.", timestamp=1.0)
        assert isinstance(analyzer.feature_history[0], np.ndarray)
        assert analyzer.feature_history[0].shape == (N_FEATURES,)
