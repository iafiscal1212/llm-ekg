<p align="center">
  <h1 align="center">LLM EKG</h1>
  <p align="center">
    <strong>Is your AI getting dumber? Now you can prove it.</strong>
  </p>
  <p align="center">
    <a href="https://zenodo.org/records/19284461"><img src="https://zenodo.org/badge/doi/10.5281/zenodo.19284461.svg" alt="DOI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSL_1.1-blue" alt="License"></a>
    <a href="https://pypi.org/project/llm-ekg"><img src="https://img.shields.io/badge/python-3.9%2B-green" alt="Python"></a>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#live-monitoring">Live Monitoring</a> &bull;
    <a href="#html-report">Report</a> &bull;
    <a href="#how-it-works">How It Works</a> &bull;
    <a href="https://zenodo.org/records/19284461">Paper</a>
  </p>
</p>

---

LLM EKG is a **mathematical health monitor** for Large Language Models. It analyzes LLM outputs as time series to detect **degradation**, **hallucination**, and **behavioral drift** — using pure mathematics, not NLP.

No embeddings. No tokenizers. No external AI. Just `numpy`.

```
RESULT: DEGRADED (74/100)
Trend: +0.1651
Hallucination risk: 22.96%
Mean persistence: 0.578
```

## Why?

Every company runs LLMs in production. Nobody monitors their **output quality** mathematically.

- GPT-4 getting lazier over time? **LLM EKG detects it.**
- Claude hallucinating more after an update? **LLM EKG catches it.**
- Your fine-tuned model degrading silently? **LLM EKG raises the alarm.**

The big labs will never build this — it exposes their problems. So we did.

## Quick Start

```bash
pip install llm-ekg
```

### One command

```bash
# Auto-detects format (ChatGPT, Claude, CSV, JSONL, plain text)
llm-ekg conversation.json

# Explicit format
llm-ekg --format chatgpt export.json -o report.html
```

### Three lines of Python

```python
from llm_ekg import LLMAnalyzer

analyzer = LLMAnalyzer()
for response in my_responses:
    result = analyzer.ingest(response["text"])

print(f"{analyzer.get_summary()['verdict']} — {analyzer.get_summary()['global_score_100']}/100")
```

## Live Monitoring

Wrap your OpenAI or Anthropic client. Zero code changes.

```python
from llm_ekg import LiveMonitor

monitor = LiveMonitor()

# OpenAI
import openai
client = monitor.wrap_openai(openai.OpenAI())

# Use exactly as before — monitoring is automatic
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

# Check health anytime
print(f"Score: {monitor.score}/100 — {monitor.verdict}")

# Generate full HTML report
monitor.report("ekg.html")
```

Works with Anthropic too:

```python
import anthropic
client = monitor.wrap_anthropic(anthropic.Anthropic())
```

## What It Detects

| Signal | Meaning |
|--------|---------|
| **Anomaly rising** | Model quality is degrading |
| **Drift spike** | Sudden behavioral shift |
| **Confidence mismatch** | Hallucination (specific claims + zero hedging) |
| **Assertion density up** | Model becoming overconfident |
| **Persistence > 0.5** | Degradation is trending, not random |
| **Persistence < 0.5** | Model self-correcting |

## How It Works

LLM EKG extracts **16 numerical features** from each response — no NLP, no language models, no semantic analysis:

**Degradation signals** (0-11): response length, word count, vocabulary diversity, word length, sentence count, sentence length, punctuation density, hedge ratio, list usage, code ratio, repetition score, latency.

**Hallucination signature** (12-15): specificity score (concrete details density), confidence mismatch (specificity vs hedging gap), assertion density (certainty vs uncertainty ratio), self-consistency (internal contradiction score).

These features feed into a proprietary **behavioral state engine** that computes anomaly scores, drift magnitude, and multi-scale persistence analysis.

**All diagnostics are data-driven** — zero hardcoded thresholds. Every metric is compared against its own distribution within the session.

## HTML Report

Self-contained HTML file. No JavaScript dependencies. Opens in any browser.

9 sections: Executive Summary, Hallucination Monitor, EKG Temporal, Behavioral Metrics (M0-M3), Drift Map, Multi-Scale Analysis, Trend Persistence, Feature Timeline, Diagnostic.

### Run the demo

```bash
git clone https://github.com/iafiscal1212/llm-ekg.git
cd llm-ekg
pip install -e .
python demo.py
# Open demo_ekg_report.html
```

## Supported Formats

| Format | Extension | Source |
|--------|-----------|--------|
| ChatGPT | `.json` | Settings → Export data |
| Claude | `.json` | claude.ai export |
| API Log | `.csv` | CSV with `response` column |
| JSONL | `.jsonl` | One JSON per line |
| Plain Text | `.txt` | Blank-line separated |

## Dependencies

`numpy` + `matplotlib`. That's it.

Optional: `openai` and/or `anthropic` for live monitoring.

```bash
pip install llm-ekg[openai]     # OpenAI wrapper
pip install llm-ekg[anthropic]  # Anthropic wrapper
pip install llm-ekg[all]        # Both
```

## License

[Business Source License 1.1](LICENSE) — the same license used by Redis, MariaDB, and Sentry.

- **Always free**: personal use, internal monitoring, research, education, open source
- **Commercial license required**: if you sell LLM monitoring as a service
- **Converts to Apache 2.0**: March 28, 2030

Contact: carmen@iafiscal.es

## Cite

If you use LLM EKG in your research, please cite:

```bibtex
@software{esteban2026llmekg,
  author    = {Esteban, Carmen},
  title     = {LLM EKG: A Mathematical Health Monitor for Large Language Models},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19284461},
  url       = {https://doi.org/10.5281/zenodo.19284461}
}
```

## Author

**Carmen Esteban** — [IAFISCAL & PARTNERS](https://github.com/iafiscal1212)
