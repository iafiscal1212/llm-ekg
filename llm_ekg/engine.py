"""
Core analysis engine for LLM EKG.

(c) 2026 Carmen Esteban — IAFISCAL & PARTNERS
Proprietary mathematical analysis. All rights reserved.
"""

import re
import numpy as np
from collections import deque
from typing import Optional


# ── Physical constants (multi-scale frequency generation) ──────────────

_MU_0 = 4 * np.pi * 1e-7
_EPS_0 = 8.854e-12
_EPS_R = 9.8
_BASE = 5e-3
_THICK = 200e-9
_WIDTH = 10e-6
_LAMBDA = 140e-9


def _lc_freq(perimeter):
    w, t = _WIDTH, _THICK
    log_arg = max(2 * perimeter / w, 2.72)
    l_g = _MU_0 * perimeter / (2 * np.pi) * (np.log(log_arg) - 0.774)
    l_k = _MU_0 * _LAMBDA**2 * perimeter / (w * t)
    l_t = l_g + l_k
    eps_eff = (_EPS_R + 1) / 2
    c = _EPS_0 * eps_eff * w / 500e-6 * perimeter
    return 1.0 / (2 * np.pi * np.sqrt(l_t * c))


def _scale_frequencies(n_gen, scale_factor=0.5, sides=3):
    freqs = []
    for k in range(n_gen):
        side = _BASE / (scale_factor ** -(k + 1))
        side = _BASE * (scale_factor ** (k + 1))
        perim = sides * side
        freqs.append(_lc_freq(perim))
    log_f = np.log(np.array(freqs))
    lo, hi = log_f[0], log_f[-1]
    if hi - lo < 1e-12:
        return np.linspace(0, 1, n_gen)
    return (log_f - lo) / (hi - lo)


# ── Feature extraction (16 features, zero NLP) ────────────────────────

FEATURE_NAMES = [
    "length_chars", "length_words", "vocab_diversity", "avg_word_length",
    "sentence_count", "avg_sentence_length", "punctuation_density",
    "hedge_ratio", "list_ratio", "code_ratio", "repetition_score",
    "response_time_s",
    "specificity_score", "confidence_mismatch", "assertion_density",
    "self_consistency",
]
N_FEATURES = len(FEATURE_NAMES)

_HEDGE = {
    "perhaps", "maybe", "might", "could", "possibly", "generally",
    "typically", "usually", "however", "although", "somewhat",
    "relatively", "approximately", "roughly", "arguably", "likely",
    "potentially", "presumably", "apparently", "supposedly",
    "quiza", "quizas", "posiblemente", "generalmente", "aproximadamente",
    "relativamente", "probablemente", "normalmente", "aparentemente",
    "supuestamente", "presumiblemente", "talvez",
}
_PUNCT = set(".,;:!?()[]{}\"'-/\\@#$%^&*+=<>~`|")
_SPEC_PAT = [
    re.compile(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b'),
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
    re.compile(r'\b(?:19|20)\d{2}\b'),
    re.compile(r'\b\d+(?:\.\d+)?%\b'),
    re.compile(r'\b\d{1,2}:\d{2}\b'),
    re.compile(r'\b(?:USD|EUR|GBP|\$|€|£)\s?\d'),
]
_ASSERT = {
    "is", "are", "was", "were", "will", "always", "never", "every",
    "must", "certainly", "definitely", "clearly", "obviously",
    "undoubtedly", "absolutely", "exactly", "precisely", "specifically",
    "es", "son", "fue", "siempre", "nunca", "cada", "todo", "todos",
    "debe", "ciertamente", "definitivamente", "claramente", "obviamente",
    "exactamente", "precisamente", "especificamente",
}
_COND = {
    "if", "unless", "whether", "might", "could", "would", "should",
    "may", "perhaps", "maybe", "possibly", "probably", "likely",
    "sometimes", "often", "rarely", "occasionally",
    "si", "salvo", "aunque", "podria", "deberia", "quiza",
    "posiblemente", "probablemente", "aveces", "raramente",
}
_NEG = {
    "not", "no", "never", "neither", "nor", "none", "nothing",
    "nowhere", "cannot", "can't", "won't", "don't", "doesn't",
    "isn't", "aren't", "wasn't", "weren't", "shouldn't", "wouldn't",
    "couldn't", "haven't", "hasn't", "hadn't",
    "nunca", "ninguno", "ninguna", "nada", "tampoco", "jamas",
}


class LLMFeatureExtractor:
    def __init__(self):
        self._prev_bi: set = set()

    def extract(self, response: str, response_time_s: float = 0.0) -> np.ndarray:
        v = np.zeros(N_FEATURES, dtype=np.float64)
        if not response or not response.strip():
            return v
        text = response.strip()
        v[0] = len(text)
        words = text.split()
        nw = len(words)
        v[1] = nw
        if nw == 0:
            return v
        wl = [w.lower().strip(".,;:!?()[]{}\"'-") for w in words]
        wl = [w for w in wl if w]
        nwl = max(len(wl), 1)
        v[2] = len(set(wl)) / nwl
        v[3] = np.mean([len(w) for w in wl]) if wl else 0.0
        sents = re.split(r'[.!?]+\s+', text)
        sents = [s for s in sents if s.strip()]
        ns = max(len(sents), 1)
        v[4] = ns
        v[5] = nw / ns
        v[6] = sum(1 for c in text if c in _PUNCT) / len(text)
        nh = sum(1 for w in wl if w in _HEDGE)
        hr = nh / nwl
        v[7] = hr
        lines = [l for l in text.split('\n') if l.strip()]
        if lines:
            nl = sum(1 for l in lines
                     if re.match(r'^\s*[-*]\s', l) or re.match(r'^\s*\d+[.)]\s', l))
            v[8] = nl / len(lines)
        cb = re.findall(r'```[\s\S]*?```', text)
        v[9] = sum(len(b) for b in cb) / len(text) if text else 0.0
        bi = set()
        for i in range(len(wl) - 1):
            bi.add((wl[i], wl[i + 1]))
        if bi and self._prev_bi:
            v[10] = len(bi & self._prev_bi) / max(len(bi | self._prev_bi), 1)
        self._prev_bi = bi
        v[11] = response_time_s
        nsp = sum(len(p.findall(text)) for p in _SPEC_PAT)
        v[12] = nsp / ns
        sn = min(v[12] / 3.0, 1.0)
        v[13] = sn * (1.0 - min(hr * 10, 1.0))
        na = sum(1 for w in wl if w in _ASSERT)
        nc = sum(1 for w in wl if w in _COND)
        v[14] = na / (na + nc) if na + nc > 0 else 0.5
        mid = len(wl) // 2
        nf = sum(1 for w in wl[:mid] if w in _NEG)
        nsec = sum(1 for w in wl[mid:] if w in _NEG)
        nd = (nf + nsec) / nwl
        ni = abs(nf - nsec) / max(nf + nsec, 1)
        v[15] = ni * 0.6 + min(nd * 5, 1.0) * 0.4
        return v


# ── Behavioral state engine ───────────────────────────────────────────

class _StateEngine:
    def __init__(self, input_dim, hidden_dim, seed=42):
        self._id = input_dim
        self._hd = hidden_dim
        self.h = np.zeros(hidden_dim, dtype=np.float64)
        self._mom = np.zeros(hidden_dim, dtype=np.float64)
        self._step = 0
        rng = np.random.RandomState(seed)
        sc = np.sqrt(2.0 / (input_dim + hidden_dim))
        self._Wi = rng.randn(hidden_dim, input_dim).astype(np.float64) * sc
        self._Wi_norm0 = float(np.linalg.norm(self._Wi))
        self._Wh = rng.randn(hidden_dim, hidden_dim).astype(np.float64) * sc
        sc2 = np.sqrt(2.0 / (hidden_dim + 4))
        self._We = rng.randn(hidden_dim, 4).astype(np.float64) * sc2
        self._Ws = rng.randn(input_dim, hidden_dim).astype(np.float64) / np.sqrt(hidden_dim)
        self._prev_h = None
        self._h_hist = []
        self._drift_hist = deque(maxlen=100)
        self._e_hist = deque(maxlen=100)
        self._wc_hist = deque(maxlen=100)
        self._en_hist = deque(maxlen=100)

    def _compute_e(self, x):
        md = min(len(x), len(self.h))
        xp, hp = x[:md], self.h[:md]
        hn = np.linalg.norm(hp)
        e0 = np.dot(xp, hp) / (np.linalg.norm(xp) * hn + 1e-10) if hn > 1e-10 else 0.0
        e1 = np.tanh(np.var(x) / 10.0)
        e2 = 0.5 + e0 * 0.1
        xa = np.abs(x) + 1e-10
        xn = xa / np.sum(xa)
        e3 = -np.sum(xn * np.log(xn + 1e-10)) / np.log(max(len(x), 2))
        return np.array([e0, e1, e2, e3], dtype=np.float64)

    def step(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        if len(x) < self._id:
            x = np.pad(x, (0, self._id - len(x)))
        elif len(x) > self._id:
            x = x[:self._id]
        e = self._compute_e(x)
        ss = self._Ws @ self.h
        cx = x + 0.15 * ss
        ic = self._Wi @ cx
        rc = self._Wh @ self.h
        ec = self._We @ e
        pt = np.random.randn(self._hd) * 0.01 * (1.0 + e[3])
        dh = ic + rc * 0.5 + ec * 0.3 + pt
        dh = np.tanh(dh) + 0.1 * np.sin(dh * 3.14159)
        self._mom = 0.9 * self._mom + 0.1 * dh
        nh = self.h + 0.1 * self._mom
        norm = np.linalg.norm(nh)
        if norm > 10.0:
            nh *= 10.0 / norm
        if np.any(np.isnan(nh)) or np.any(np.isinf(nh)):
            nh = self.h
        drift = float(np.linalg.norm(nh - self._prev_h)) if self._prev_h is not None else 0.0
        self._prev_h = self.h.copy()
        self.h = nh
        self._h_hist.append(self.h.copy())
        if len(self._h_hist) > 10:
            self._h_hist.pop(0)
        hc = 0.0
        if len(self._h_hist) >= 10:
            prev_w = self._Wi.copy()
            hs = np.array(self._h_hist)
            vh = float(np.mean(np.var(hs, axis=0)))
            eta = vh / self._hd
            self._Wi += eta * np.outer(self.h, cx)
            wn = np.linalg.norm(self._Wi)
            if wn > self._Wi_norm0:
                self._Wi *= self._Wi_norm0 / wn
            hc = float(np.linalg.norm(self._Wi - prev_w)) / (self._Wi_norm0 + 1e-10)
        self._wc_hist.append(hc)
        self._drift_hist.append(drift)
        self._e_hist.append(e.copy())
        self._step += 1
        er = sum(self._drift_hist) / max(self._step, 1)
        self._en_hist.append(er)
        sn = np.linalg.norm(ss)
        wn = np.linalg.norm(x)
        sr = sn / (wn + 1e-10)
        anom = self._anomaly(drift, e, hc) if self._step >= 20 else 0.0
        return {
            "metrics": e.tolist(), "m0_memory": float(e[0]),
            "m1_variability": float(e[1]), "m2_persistence": float(e[2]),
            "m3_complexity": float(e[3]), "drift": drift,
            "state_change": hc, "self_ratio": sr, "energy_rate": er,
            "anomaly_score": anom, "h_norm": float(np.linalg.norm(self.h)),
            "step": self._step,
        }

    def _anomaly(self, drift, e, hc):
        sc = []
        if self._drift_hist:
            d = np.array(self._drift_hist)
            z = abs(drift - d.mean()) / (d.std() + 1e-10)
            sc.append(min(z / 3.0, 1.0) * 0.35)
        if self._e_hist:
            e3s = np.array([x[3] for x in self._e_hist])
            z = (e[3] - e3s.mean()) / (e3s.std() + 1e-10)
            sc.append(max(min(z / 3.0, 1.0), 0.0) * 0.25)
            e0s = np.array([x[0] for x in self._e_hist])
            drop = max(e0s.mean() - e[0], 0.0)
            sc.append(min(drop / 0.5, 1.0) * 0.20)
        if self._wc_hist:
            wc = np.array(self._wc_hist)
            z = (hc - wc.mean()) / (wc.std() + 1e-10)
            sc.append(max(min(z / 3.0, 1.0), 0.0) * 0.20)
        return sum(sc) if sc else 0.0


# ── Multi-scale frequency analyzer ───────────────────────────────────

class _FreqAnalyzer:
    def __init__(self, n_gen=6, fft_window=32):
        self.n_gen = n_gen
        self.fft_window = fft_window
        self._band_pos = _scale_frequencies(n_gen)

    def analyze(self, series):
        n = len(series)
        if n < self.fft_window:
            return {"band_energies": np.zeros(self.n_gen), "hurst": 0.5}
        s = series[-self.fft_window:]
        w = np.hanning(len(s))
        fv = np.fft.rfft(s * w)
        pw = np.abs(fv) ** 2
        pw[0] = 0
        nb = len(pw)
        cb = 1 + self._band_pos * (nb - 2)
        ng = self.n_gen
        bw = np.zeros(ng)
        for i in range(ng):
            if ng == 1:
                bw[i] = nb / 2
            elif i == 0:
                bw[i] = (cb[1] - cb[0]) / 2
            elif i == ng - 1:
                bw[i] = (cb[-1] - cb[-2]) / 2
            else:
                bw[i] = min(cb[i] - cb[i-1], cb[i+1] - cb[i]) / 2
            bw[i] = max(bw[i], 1.0)
        bi = np.arange(nb, dtype=np.float64)
        en = np.zeros(ng)
        for i in range(ng):
            g = np.exp(-0.5 * ((bi - cb[i]) / bw[i]) ** 2)
            en[i] = np.sum(pw * g)
        mx = en.max() + 1e-12
        be = en / mx
        le = np.log(en + 1e-15)
        sl, _ = np.polyfit(np.arange(ng, dtype=np.float64), le, 1)
        h = float(np.clip(0.5 - 0.5 * np.tanh(sl / 2.0), 0.01, 0.99))
        return {"band_energies": be, "hurst": h}

    def analyze_all(self, fh):
        ns, nf = fh.shape
        res = [self.analyze(fh[:, i]) for i in range(nf)]
        hs = [r["hurst"] for r in res]
        return {"per_feature": res, "mean_hurst": float(np.mean(hs)),
                "hursts": hs, "hurst_labels": [_hl(h) for h in hs]}


def _hl(h):
    if h > 0.65: return "Trending"
    elif h > 0.55: return "Persistent"
    elif h >= 0.45: return "Random"
    elif h >= 0.35: return "Mean-Revert"
    else: return "Anti-Persist"


# ── Main analyzer ─────────────────────────────────────────────────────

class LLMAnalyzer:
    """Mathematical health analyzer for LLM outputs."""

    def __init__(self, n_scales=6, fft_window=32):
        self.extractor = LLMFeatureExtractor()
        self._engine = _StateEngine(input_dim=N_FEATURES, hidden_dim=48, seed=42)
        self._freq = _FreqAnalyzer(n_gen=n_scales, fft_window=fft_window)
        self.feature_history: list[np.ndarray] = []
        self.state_history: list[dict] = []
        self.scale_history: list[Optional[dict]] = []
        self.timestamps: list[float] = []
        self.metadata: list[dict] = []

    def ingest(self, response: str, timestamp: float = 0.0,
               response_time_s: float = 0.0, **extra) -> dict:
        features = self.extractor.extract(response, response_time_s)
        self.feature_history.append(features)
        self.timestamps.append(timestamp)
        self.metadata.append(extra)
        ir = self._engine.step(features)
        self.state_history.append(ir)
        fr = None
        if len(self.feature_history) >= self._freq.fft_window:
            fr = self._freq.analyze_all(np.array(self.feature_history))
        self.scale_history.append(fr)
        return {"step": len(self.feature_history), "features": dict(zip(FEATURE_NAMES, features)),
                "state": ir, "multiscale": fr}

    def ingest_batch(self, responses: list[dict]) -> list[dict]:
        return [self.ingest(r.get("response", ""), r.get("timestamp", 0.0),
                            r.get("response_time_s", 0.0), model=r.get("model", ""))
                for r in responses]

    def get_summary(self) -> dict:
        n = len(self.state_history)
        if n == 0:
            return {"n_responses": 0, "verdict": "NO DATA"}
        scores = [h["anomaly_score"] for h in self.state_history]
        drifts = [h["drift"] for h in self.state_history]
        e0s = [h["m0_memory"] for h in self.state_history]
        e3s = [h["m3_complexity"] for h in self.state_history]
        tail = max(1, n // 3)
        gs = float(np.mean(scores[-tail:]))
        trend = 0.0
        if n >= 10:
            half = n // 2
            trend = float(np.mean(scores[half:])) - float(np.mean(scores[:half]))
        s100 = min(100, max(0, int((1.0 - gs) * 100)))
        verdict = "HEALTHY" if s100 >= 80 else ("DEGRADED" if s100 >= 50 else "CRITICAL")
        lf = None
        for fh in reversed(self.scale_history):
            if fh is not None:
                lf = fh
                break
        fh_arr = np.array(self.feature_history)
        cm = float(fh_arr[:, 13].mean())
        ad = float(fh_arr[:, 14].mean())
        sc = float(fh_arr[:, 15].mean())
        hr = min(cm * 0.40 + min(ad, 1.0) * 0.30 + sc * 0.30, 1.0)
        if n >= 5:
            rhr = float(fh_arr[-tail:, 13].mean()) * 0.40 + \
                  min(float(fh_arr[-tail:, 14].mean()), 1.0) * 0.30 + \
                  float(fh_arr[-tail:, 15].mean()) * 0.30
        else:
            rhr = hr
        return {
            "n_responses": n, "global_score_100": s100,
            "global_anomaly_mean": float(np.mean(scores)),
            "recent_anomaly_mean": gs, "trend": trend, "verdict": verdict,
            "drift_mean": float(np.mean(drifts)),
            "m0_mean": float(np.mean(e0s)), "m3_mean": float(np.mean(e3s)),
            "multiscale": lf,
            "specificity_mean": float(fh_arr[:, 12].mean()),
            "confidence_mismatch_mean": cm, "assertion_density_mean": ad,
            "self_consistency_mean": sc,
            "hallucination_risk": hr, "recent_hallucination_risk": rhr,
        }
