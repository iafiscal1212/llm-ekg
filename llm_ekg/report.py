"""
HTML report generator for LLM EKG.

Generates self-contained HTML with inline matplotlib charts (base64 PNG).
No external JS dependencies. Opens in any browser.

(c) 2026 Carmen Esteban — IAFISCAL & PARTNERS
"""

import base64
import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .engine import FEATURE_NAMES, N_FEATURES

# Dark palette (medical monitor style)
_BG = '#0f172a'
_FG = '#e2e8f0'
_GREEN = '#22c55e'
_RED = '#ef4444'
_YELLOW = '#eab308'
_BLUE = '#3b82f6'
_CYAN = '#06b6d4'
_PURPLE = '#a855f7'
_ORANGE = '#f97316'
_GRID = '#1e293b'


def _fig_to_base64(fig, dpi=100) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=_BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _setup_ax(ax, title=""):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.spines['bottom'].set_color(_GRID)
    ax.spines['left'].set_color(_GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color=_FG)
    if title:
        ax.set_title(title, color=_FG, fontsize=10, fontweight='bold', pad=8)


class EKGReport:
    """Generates self-contained HTML report from LLMAnalyzer results."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def generate(self, output_path: str = "llm_ekg_report.html"):
        a = self.analyzer
        summary = a.get_summary()

        sections = []
        sections.append(self._section_summary(summary))
        sections.append(self._section_hallucination(summary))
        sections.append(self._section_ekg_temporal())
        sections.append(self._section_behavioral_metrics())
        sections.append(self._section_drift())
        sections.append(self._section_multiscale())
        sections.append(self._section_persistence())
        sections.append(self._section_features())
        sections.append(self._section_diagnostic(summary))

        html = self._build_html(sections, summary)
        from pathlib import Path
        Path(output_path).write_text(html, encoding='utf-8')

    def _section_summary(self, summary: dict) -> str:
        score = summary['global_score_100']
        verdict = summary['verdict']
        n = summary['n_responses']

        if verdict == "HEALTHY":
            color = _GREEN
        elif verdict == "DEGRADED":
            color = _YELLOW
        else:
            color = _RED

        h_risk = summary.get("hallucination_risk", 0.0)
        h_recent = summary.get("recent_hallucination_risk", h_risk)
        h_pct = int(h_risk * 100)
        if h_risk > 1e-6:
            h_trend = h_recent / h_risk
        else:
            h_trend = 1.0
        if h_trend > 1.3:
            h_color, h_label = _RED, "RISING"
        elif h_trend < 0.7:
            h_color, h_label = _GREEN, "FALLING"
        else:
            h_color = _YELLOW if h_pct > 0 else _GREEN
            h_label = "STABLE"

        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}; font-size:48px">{score}</div>
                    <div class="metric-label">Score / 100</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{verdict}</div>
                    <div class="metric-label">Verdict</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color:{h_color}">{h_pct}%</div>
                    <div class="metric-label">Hallucination Risk ({h_label})</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n}</div>
                    <div class="metric-label">Responses</div>
                </div>
            </div>
        </div>
        """

    def _section_hallucination(self, summary: dict) -> str:
        fh = np.array(self.analyzer.feature_history)
        n_steps = fh.shape[0]
        steps = list(range(1, n_steps + 1))

        specificity = fh[:, 12]
        conf_mismatch = fh[:, 13]
        assertion_d = fh[:, 14]
        self_consist = fh[:, 15]
        halluc_risk = conf_mismatch * 0.40 + np.minimum(assertion_d, 1.0) * 0.30 + self_consist * 0.30

        fig, axes = plt.subplots(2, 1, figsize=(14, 6), facecolor=_BG,
                                  gridspec_kw={'height_ratios': [2, 1.2]})

        ax1 = axes[0]
        _setup_ax(ax1, "Hallucination Risk")
        ax1.fill_between(steps, halluc_risk, alpha=0.15, color=_RED)
        ax1.plot(steps, halluc_risk, color=_RED, linewidth=1.2)
        ax1.set_ylabel('Risk', color=_FG, fontsize=8)
        ax1.set_ylim(-0.02, max(halluc_risk.max() * 1.2, 0.3))
        for i, hr in enumerate(halluc_risk):
            if hr > 0.5:
                ax1.plot(i + 1, hr, 'o', color=_RED, markersize=4, alpha=0.9)
            elif hr > 0.25:
                ax1.plot(i + 1, hr, 'o', color=_YELLOW, markersize=2, alpha=0.6)

        ax2 = axes[1]
        _setup_ax(ax2, "Components")
        ax2.plot(steps, conf_mismatch, color=_RED, linewidth=0.8, alpha=0.8, label='Confidence Mismatch')
        ax2.plot(steps, assertion_d, color=_ORANGE, linewidth=0.8, alpha=0.8, label='Assertion Density')
        ax2.plot(steps, self_consist, color=_PURPLE, linewidth=0.8, alpha=0.8, label='Self-Inconsistency')
        ax2.plot(steps, specificity, color=_CYAN, linewidth=0.8, alpha=0.5, label='Specificity')
        ax2.legend(fontsize=7, loc='upper right', framealpha=0.3,
                   labelcolor=_FG, facecolor=_BG, edgecolor=_GRID)
        ax2.set_xlabel('Response #', color=_FG, fontsize=8)

        fig.tight_layout(pad=1.0)
        img = _fig_to_base64(fig)

        cm = summary.get("confidence_mismatch_mean", 0.0)
        ad = summary.get("assertion_density_mean", 0.0)
        sc = summary.get("self_consistency_mean", 0.0)
        sp = summary.get("specificity_mean", 0.0)

        return f"""
        <div class="section">
            <h2>Hallucination Monitor</h2>
            <div class="summary-grid">
                <div class="metric-card">
                    <div class="metric-value" style="color:{_RED if cm > 0.3 else _GREEN}">{cm:.3f}</div>
                    <div class="metric-label">Confidence Mismatch</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color:{_ORANGE if ad > 0.7 else _GREEN}">{ad:.3f}</div>
                    <div class="metric-label">Assertion Density</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color:{_PURPLE if sc > 0.4 else _GREEN}">{sc:.3f}</div>
                    <div class="metric-label">Self-Inconsistency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sp:.2f}</div>
                    <div class="metric-label">Specificity / sentence</div>
                </div>
            </div>
            <img src="data:image/png;base64,{img}" class="chart">
        </div>
        """

    def _section_ekg_temporal(self) -> str:
        scores = [h["anomaly_score"] for h in self.analyzer.state_history]
        steps = list(range(1, len(scores) + 1))

        fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=_BG)
        _setup_ax(ax, "EKG — Anomaly Score")
        ax.fill_between(steps, scores, alpha=0.15, color=_GREEN)
        ax.plot(steps, scores, color=_GREEN, linewidth=1.2)
        for i, s in enumerate(scores):
            if s > 0.5:
                ax.plot(i + 1, s, 'o', color=_RED, markersize=3, alpha=0.8)
            elif s > 0.3:
                ax.plot(i + 1, s, 'o', color=_YELLOW, markersize=2, alpha=0.6)
        ax.set_xlabel('Response #', color=_FG, fontsize=8)
        ax.set_ylabel('Anomaly Score', color=_FG, fontsize=8)
        ax.set_ylim(-0.02, max(max(scores) * 1.2, 0.1))

        img = _fig_to_base64(fig)
        return f"""
        <div class="section">
            <h2>EKG Temporal</h2>
            <img src="data:image/png;base64,{img}" class="chart">
        </div>
        """

    def _section_behavioral_metrics(self) -> str:
        hist = self.analyzer.state_history
        steps = list(range(1, len(hist) + 1))

        e0 = [h["m0_memory"] for h in hist]
        e1 = [h["m1_variability"] for h in hist]
        e2 = [h["m2_persistence"] for h in hist]
        e3 = [h["m3_complexity"] for h in hist]

        fig, axes = plt.subplots(2, 2, figsize=(14, 6), facecolor=_BG)
        series = [
            (e0, "M0 — Memory", _CYAN),
            (e1, "M1 — Variability", _BLUE),
            (e2, "M2 — Persistence", _PURPLE),
            (e3, "M3 — Complexity", _ORANGE),
        ]
        for ax, (data, title, color) in zip(axes.flat, series):
            _setup_ax(ax, title)
            ax.plot(steps, data, color=color, linewidth=1.0)
            ax.fill_between(steps, data, alpha=0.1, color=color)
            ax.set_xlabel('Response #', color=_FG, fontsize=7)

        fig.tight_layout(pad=1.5)
        img = _fig_to_base64(fig)

        return f"""
        <div class="section">
            <h2>Behavioral Metrics</h2>
            <img src="data:image/png;base64,{img}" class="chart">
            <p class="caption">M0: temporal memory. M1: variability. M2: persistence. M3: complexity.</p>
        </div>
        """

    def _section_drift(self) -> str:
        drifts = [h["drift"] for h in self.analyzer.state_history]
        steps = list(range(1, len(drifts) + 1))

        fig, ax = plt.subplots(figsize=(14, 3), facecolor=_BG)
        _setup_ax(ax, "Drift — State Change Magnitude")

        d_arr = np.array(drifts)
        d_mean, d_std = d_arr.mean(), d_arr.std()
        colors = []
        for d in drifts:
            if d > d_mean + 2 * d_std:
                colors.append(_RED)
            elif d > d_mean + d_std:
                colors.append(_YELLOW)
            else:
                colors.append(_CYAN)

        ax.bar(steps, drifts, color=colors, width=1.0, alpha=0.7)
        ax.axhline(y=d_mean, color=_FG, linestyle=':', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Response #', color=_FG, fontsize=8)
        ax.set_ylabel('Drift', color=_FG, fontsize=8)

        img = _fig_to_base64(fig)
        return f"""
        <div class="section">
            <h2>Drift Map</h2>
            <img src="data:image/png;base64,{img}" class="chart">
        </div>
        """

    def _section_multiscale(self) -> str:
        valid_fractals = [
            (i, f) for i, f in enumerate(self.analyzer.scale_history)
            if f is not None
        ]
        if not valid_fractals:
            return """
            <div class="section">
                <h2>Multi-Scale Analysis</h2>
                <p>Insufficient data for multi-scale analysis (requires >= 32 responses).</p>
            </div>
            """

        _, last_frac = valid_fractals[-1]
        per_feat = last_frac["per_feature"]
        n_feat = len(per_feat)
        n_gen = len(per_feat[0]["band_energies"])
        matrix = np.zeros((n_feat, n_gen))
        for i, pf in enumerate(per_feat):
            matrix[i, :] = pf["band_energies"]

        fig, ax = plt.subplots(figsize=(14, 5), facecolor=_BG)
        _setup_ax(ax, "Multi-Scale Energy Distribution")

        cmap = LinearSegmentedColormap.from_list("ekg", [_BG, _BLUE, _CYAN, _GREEN, _YELLOW])
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')

        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
        ax.set_xticks(range(n_gen))
        ax.set_xticklabels([f"Scale {i}" for i in range(n_gen)], fontsize=7)
        ax.set_xlabel('Analysis Scale', color=_FG, fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.ax.tick_params(colors=_FG, labelsize=7)

        img = _fig_to_base64(fig)
        return f"""
        <div class="section">
            <h2>Multi-Scale Analysis</h2>
            <img src="data:image/png;base64,{img}" class="chart">
        </div>
        """

    def _section_persistence(self) -> str:
        valid_fractals = [f for f in self.analyzer.scale_history if f is not None]
        if not valid_fractals:
            return """
            <div class="section">
                <h2>Trend Persistence</h2>
                <p>Insufficient data to compute persistence.</p>
            </div>
            """

        last = valid_fractals[-1]
        hursts = last["hursts"]
        labels = last["hurst_labels"]

        fig, ax = plt.subplots(figsize=(14, 4), facecolor=_BG)
        _setup_ax(ax, "Trend Persistence Index")

        colors = []
        for h in hursts:
            if h > 0.55:
                colors.append(_RED)
            elif h >= 0.45:
                colors.append(_GREEN)
            else:
                colors.append(_BLUE)

        ax.barh(range(len(hursts)), hursts, color=colors, alpha=0.8)
        ax.set_yticks(range(len(hursts)))
        ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
        ax.axvline(x=0.5, color=_FG, linestyle=':', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Persistence Index', color=_FG, fontsize=8)
        ax.set_xlim(0, 1)

        for i, (h, label) in enumerate(zip(hursts, labels)):
            ax.text(h + 0.02, i, f"{h:.2f} ({label})", va='center',
                    color=_FG, fontsize=7)

        img = _fig_to_base64(fig)
        return f"""
        <div class="section">
            <h2>Trend Persistence</h2>
            <img src="data:image/png;base64,{img}" class="chart">
            <p class="caption">Red: persistent trend. Green: random (healthy). Blue: self-correcting.</p>
        </div>
        """

    def _section_features(self) -> str:
        fh = np.array(self.analyzer.feature_history)
        n_steps = fh.shape[0]
        steps = list(range(1, n_steps + 1))

        fig, axes = plt.subplots(4, 4, figsize=(14, 10), facecolor=_BG)
        palette = [_GREEN, _CYAN, _BLUE, _PURPLE, _ORANGE, _YELLOW,
                   _RED, _GREEN, _CYAN, _BLUE, _PURPLE, _ORANGE,
                   _RED, _ORANGE, _PURPLE, _CYAN]

        for i, (ax, name, color) in enumerate(zip(axes.flat, FEATURE_NAMES, palette)):
            _setup_ax(ax, name)
            ax.plot(steps, fh[:, i], color=color, linewidth=0.8)
            ax.fill_between(steps, fh[:, i], alpha=0.08, color=color)
            ax.tick_params(labelsize=6)

        fig.tight_layout(pad=1.0)
        img = _fig_to_base64(fig)

        return f"""
        <div class="section">
            <h2>Feature Timeline</h2>
            <img src="data:image/png;base64,{img}" class="chart">
        </div>
        """

    def _section_diagnostic(self, summary: dict) -> str:
        """Data-driven diagnostic. Zero hardcoded thresholds.
        All comparisons against the session's own distribution."""
        diag_lines = []
        hist = self.analyzer.state_history
        fh = np.array(self.analyzer.feature_history)
        n = len(hist)

        if n < 10:
            return """
            <div class="section">
                <h2>Diagnostic</h2>
                <p>Insufficient data for diagnostic (minimum 10 responses).</p>
            </div>
            """

        half = n // 2
        scores = np.array([h["anomaly_score"] for h in hist])
        first_mean = scores[:half].mean()
        second_mean = scores[half:].mean()
        trend_ratio = second_mean / max(first_mean, 1e-10)

        if trend_ratio > 1.5:
            diag_lines.append(
                f'<li class="warn">Rising anomaly: second half of session '
                f'has {trend_ratio:.1f}x more anomaly than the first.</li>')
        elif trend_ratio < 0.67:
            diag_lines.append(
                f'<li class="good">Decreasing anomaly: second half of session '
                f'has {1/trend_ratio:.1f}x less anomaly than the first.</li>')
        else:
            diag_lines.append(
                f'<li class="neutral">Stable anomaly throughout session '
                f'(ratio 2nd/1st half: {trend_ratio:.2f}).</li>')

        e0s = np.array([h["m0_memory"] for h in hist])
        e0_mean, e0_std = e0s.mean(), e0s.std()
        e0_cv = e0_std / max(abs(e0_mean), 1e-10)
        if e0_cv > 1.0:
            diag_lines.append(
                f'<li class="warn">Highly variable coherence (CV={e0_cv:.2f}): '
                f'responses fluctuate strongly.</li>')
        elif e0_cv > 0.5:
            diag_lines.append(
                f'<li class="neutral">Variable coherence (CV={e0_cv:.2f}): '
                f'normal diversity in multi-task sessions.</li>')
        else:
            diag_lines.append(
                f'<li class="good">Stable coherence (CV={e0_cv:.2f}): '
                f'consistent responses.</li>')

        e3s = np.array([h["m3_complexity"] for h in hist])
        e3_first = e3s[:half].mean()
        e3_second = e3s[half:].mean()
        e3_change = (e3_second - e3_first) / max(e3_first, 1e-10)
        if e3_change > 0.2:
            diag_lines.append(
                f'<li class="warn">Rising complexity: +{e3_change*100:.0f}% in the '
                f'second half of the session.</li>')
        elif e3_change < -0.2:
            diag_lines.append(
                f'<li class="good">Decreasing complexity: {e3_change*100:.0f}% in the '
                f'second half.</li>')
        else:
            diag_lines.append(
                '<li class="neutral">Stable complexity throughout session.</li>')

        if summary.get("multiscale") and summary["multiscale"].get("mean_hurst"):
            mh = summary["multiscale"]["mean_hurst"]
            if mh > 0.5:
                diag_lines.append(
                    f'<li class="neutral">Persistence: {mh:.3f} — changing metrics '
                    f'tend to keep changing in the same direction.</li>')
            else:
                diag_lines.append(
                    f'<li class="neutral">Persistence: {mh:.3f} — changing metrics '
                    f'tend to revert toward their mean.</li>')

        cm = fh[:, 13]
        cm_mean, cm_std = cm.mean(), cm.std()
        n_high_cm = int(np.sum(cm > cm_mean + 2 * cm_std))
        pct_high = n_high_cm / n * 100

        ad = fh[:, 14]
        ad_first = ad[:half].mean()
        ad_second = ad[half:].mean()
        ad_trend = (ad_second - ad_first) / max(ad_first, 1e-10)

        if n_high_cm > 0:
            diag_lines.append(
                f'<li class="warn">Hallucination: {n_high_cm} responses ({pct_high:.0f}%) with '
                f'anomalous certainty/specificity mismatch (>2\u03c3 from baseline).</li>')
        else:
            diag_lines.append(
                '<li class="good">Hallucination: no responses with anomalous '
                'certainty/specificity mismatch.</li>')

        if abs(ad_trend) > 0.1:
            direction = "rising" if ad_trend > 0 else "falling"
            cls = "warn" if ad_trend > 0 else "good"
            diag_lines.append(
                f'<li class="{cls}">Assertiveness {direction}: '
                f'{ad_trend*100:+.0f}% in the second half of the session.</li>')

        items = "\n".join(diag_lines)
        return f"""
        <div class="section">
            <h2>Diagnostic</h2>
            <ul class="diagnostic">
                {items}
            </ul>
            <p class="caption">Each metric is compared against its own distribution within the session.</p>
        </div>
        """

    def _build_html(self, sections: list, summary: dict) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        verdict = summary["verdict"]
        score = summary["global_score_100"]
        body = "\n".join(sections)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM EKG — Score {score}/100 ({verdict})</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: {_BG};
    color: {_FG};
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px;
    line-height: 1.5;
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}}
h1 {{ font-size: 24px; margin-bottom: 4px; color: {_GREEN}; letter-spacing: 2px; }}
h2 {{ font-size: 16px; margin-bottom: 12px; color: {_CYAN}; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }}
.header {{ text-align: center; margin-bottom: 30px; padding: 20px; border: 1px solid #1e293b; border-radius: 8px; }}
.header .subtitle {{ color: #64748b; font-size: 11px; }}
.section {{ margin-bottom: 30px; padding: 16px; border: 1px solid #1e293b; border-radius: 8px; background: #0f172a; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.metric-card {{ text-align: center; padding: 16px 8px; border: 1px solid #1e293b; border-radius: 6px; background: #1e293b; }}
.metric-value {{ font-size: 28px; font-weight: bold; color: {_FG}; }}
.metric-label {{ font-size: 10px; color: #64748b; margin-top: 4px; }}
.chart {{ width: 100%; border-radius: 4px; margin: 8px 0; }}
.caption {{ font-size: 10px; color: #64748b; margin-top: 4px; }}
.diagnostic {{ list-style: none; padding: 0; }}
.diagnostic li {{ padding: 8px 12px; margin: 4px 0; border-radius: 4px; border-left: 3px solid; }}
.diagnostic li.good {{ border-color: {_GREEN}; background: rgba(34, 197, 94, 0.05); }}
.diagnostic li.warn {{ border-color: {_RED}; background: rgba(239, 68, 68, 0.05); }}
.diagnostic li.neutral {{ border-color: #64748b; background: rgba(100, 116, 139, 0.05); }}
.footer {{ text-align: center; color: #475569; font-size: 10px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #1e293b; }}
@media (max-width: 768px) {{ .summary-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<div class="header">
    <h1>LLM EKG</h1>
    <div class="subtitle">Mathematical Health Monitor for LLMs — {now}</div>
</div>

{body}

<div class="footer">
    <p>LLM EKG v1.0 — Carmen Esteban / IAFISCAL &amp; PARTNERS</p>
    <p>Generated: {now}</p>
</div>
</body>
</html>"""
