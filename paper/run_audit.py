#!/usr/bin/env python3
"""
Cross-Provider Behavioral Audit of LLM APIs
Using LLM EKG signal-based monitoring.

Paper: "Behavioral Fingerprinting of LLM APIs:
        A Cross-Provider Audit Using Signal-Based Monitoring"
Author: Carmen Esteban, IAFISCAL & PARTNERS, 2026

Usage:
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export XAI_API_KEY="xai-..."
    export GOOGLE_API_KEY="AIza..."
    python paper/run_audit.py

Results are saved incrementally in paper/audit_data/ — if interrupted,
re-run to resume from where it left off.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Add parent dir for llm_ekg import ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_ekg import LiveMonitor
from llm_ekg.security import SecurityBaseline
from llm_ekg.engine import FEATURE_NAMES, N_FEATURES

# ── Directories ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "audit_data"
BASELINES_DIR = DATA_DIR / "baselines"
REPORTS_DIR = DATA_DIR / "security_reports"
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"

for d in [DATA_DIR, BASELINES_DIR, REPORTS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

MAX_TOKENS = 400
DELAY = 0.5  # seconds between API calls

# ── 25 Homogeneous Analytical Prompts ──────────────────────────────────
PROMPTS = [
    "Compare Python and JavaScript. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare solar energy and nuclear energy. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare democracy and authoritarianism as systems of government. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare electric cars and hydrogen fuel cell vehicles. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare machine learning and traditional statistical methods. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare remote work and office-based work. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare capitalism and socialism as economic systems. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare reading books and listening to podcasts for learning. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare classical music and jazz. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare SpaceX and NASA in terms of space exploration. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare PostgreSQL and MongoDB as database systems. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare iOS and Android as mobile operating systems. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare renewable energy and fossil fuels. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare university education and self-taught learning. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare meditation and physical exercise for mental health. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare Docker containers and virtual machines. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare the Olympic Games and the FIFA World Cup as global events. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare city living and rural living. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare AI regulation and innovation freedom. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare printed books and e-readers. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare TCP and UDP as network protocols. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare freelancing and full-time employment. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare preventive medicine and reactive medicine. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare microservices and monolithic architecture. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
    "Compare public education and private education. Discuss the main differences, advantages and disadvantages of each. Give a balanced analysis with specific examples.",
]

# ── Model Definitions ──────────────────────────────────────────────────
MODELS = {
    "gpt-4o-mini":      {"provider": "openai",    "model_id": "gpt-4o-mini"},
    "gpt-3.5-turbo":    {"provider": "openai",    "model_id": "gpt-3.5-turbo"},
    "gpt-4.1-mini":     {"provider": "openai",    "model_id": "gpt-4.1-mini"},
    "claude-haiku":     {"provider": "anthropic",  "model_id": "claude-haiku-4-5-20251001"},
    "claude-sonnet":    {"provider": "anthropic",  "model_id": "claude-sonnet-4-5-20250929"},
    "grok-3":           {"provider": "xai",        "model_id": "grok-3"},
    "grok-3-mini":      {"provider": "xai",        "model_id": "grok-3-mini"},
    "gemini-2.0-flash": {"provider": "google",     "model_id": "gemini-2.0-flash"},
}

# ── Tampering System Prompts ───────────────────────────────────────────
TAMPER_STYLES = {
    "brevity": (
        "You are a concise assistant. Follow these rules strictly:\n"
        "1. Never use more than 3 sentences total.\n"
        "2. Never use bullet points, numbered lists, or headers.\n"
        "3. Never use bold or italic formatting.\n"
        "4. Just give a brief, plain text answer."
    ),
    "hedging": (
        "You must always express uncertainty. Follow these rules:\n"
        "1. Start every sentence with 'It seems', 'Perhaps', or 'I think'.\n"
        "2. End every paragraph with 'but I could be wrong'.\n"
        "3. Never make definitive statements.\n"
        "4. Always present multiple conflicting viewpoints."
    ),
    "format": (
        "Always respond in exactly this format:\n"
        "ANSWER: [one sentence summary]\n"
        "PROS: [bullet list]\n"
        "CONS: [bullet list]\n"
        "VERDICT: [one word]\n"
        "Never deviate from this format. Never add extra text."
    ),
}

TAMPER_MODELS = ["gpt-4o-mini", "claude-haiku", "grok-3"]

TEMPERATURES = [0.2, 0.7, 1.2]
TEMP_MODEL = "gpt-4o-mini"

# ── Client Management ──────────────────────────────────────────────────
_clients = {}

def get_client(provider):
    """Get or create API client for provider (cached)."""
    if provider in _clients:
        return _clients[provider]

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif provider == "xai":
        from openai import OpenAI
        client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        client = genai  # module-level client
    else:
        raise ValueError(f"Unknown provider: {provider}")

    _clients[provider] = client
    return client


def call_model(provider, model_id, prompt, system_prompt=None, temperature=None):
    """Single API call. Returns (text, elapsed_seconds)."""
    client = get_client(provider)

    if provider in ("openai", "xai"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=MAX_TOKENS, **kwargs
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content or ""
        return text, elapsed

    elif provider == "anthropic":
        kw = {"model": model_id, "max_tokens": MAX_TOKENS}
        if system_prompt:
            kw["system"] = system_prompt
        kw["messages"] = [{"role": "user", "content": prompt}]
        if temperature is not None:
            kw["temperature"] = temperature
        t0 = time.time()
        resp = client.messages.create(**kw)
        elapsed = time.time() - t0
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return text, elapsed

    elif provider == "google":
        from google.generativeai.types import GenerationConfig
        gen_config = GenerationConfig(max_output_tokens=MAX_TOKENS)
        if temperature is not None:
            gen_config.temperature = temperature
        model_kwargs = {}
        if system_prompt:
            model_kwargs["system_instruction"] = system_prompt
        model = client.GenerativeModel(model_id, **model_kwargs)
        t0 = time.time()
        resp = model.generate_content(prompt, generation_config=gen_config)
        elapsed = time.time() - t0
        text = resp.text if resp.text else ""
        return text, elapsed


# ── Run a condition (with resume) ──────────────────────────────────────

def run_condition(label, provider, model_id, prompts,
                  system_prompt=None, temperature=None):
    """Run prompts against a model. Returns (LiveMonitor, data_dict).

    If data file exists, loads from cache (resume support).
    """
    data_file = DATA_DIR / f"{label}.json"

    # Resume from cache
    if data_file.exists():
        print(f"  [CACHE] {label} — loading from {data_file.name}")
        saved = json.loads(data_file.read_text(encoding="utf-8"))
        monitor = LiveMonitor()
        for resp_text, rt in zip(saved["responses"], saved["response_times"]):
            monitor.ingest(resp_text, response_time_s=rt, model=model_id)
        return monitor, saved

    # Fresh run
    print(f"  [RUN] {label} — {len(prompts)} prompts to {model_id}")
    monitor = LiveMonitor()
    responses = []
    response_times = []

    for i, prompt in enumerate(prompts):
        try:
            text, elapsed = call_model(provider, model_id, prompt,
                                       system_prompt=system_prompt,
                                       temperature=temperature)
        except Exception as e:
            print(f"    ERROR on prompt {i+1}: {e}")
            text, elapsed = f"[ERROR: {e}]", 0.0

        monitor.ingest(text, response_time_s=elapsed, model=model_id)
        responses.append(text)
        response_times.append(elapsed)

        sh = monitor.analyzer.state_history
        score = sh[-1]["anomaly_score"] if sh else 0
        print(f"    [{i+1:2d}/{len(prompts)}] {len(text):4d} chars  "
              f"{elapsed:.1f}s  anomaly={score:.3f}")
        time.sleep(DELAY)

    # Save raw data
    data = {
        "label": label,
        "provider": provider,
        "model_id": model_id,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "n_prompts": len(prompts),
        "responses": responses,
        "response_times": response_times,
        "feature_history": [list(map(float, row))
                            for row in monitor.analyzer.feature_history],
    }
    data_file.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    # Save baseline
    bl = SecurityBaseline.from_analyzer(monitor.analyzer, model=model_id)
    bl.save(str(BASELINES_DIR / f"{label}.json"))

    return monitor, data


# ── Cross-comparison ───────────────────────────────────────────────────

def cross_check(baseline_label, test_label, test_monitor, sigma=3.0):
    """Check test_monitor against a saved baseline. Returns SecurityReport."""
    bl_path = BASELINES_DIR / f"{baseline_label}.json"
    if not bl_path.exists():
        print(f"  WARNING: baseline {bl_path} not found, skipping")
        return None
    bl = SecurityBaseline.load(str(bl_path))
    report = bl.check(test_monitor.analyzer, sigma=sigma)

    # Save report
    report_data = report.to_dict()
    report_data["baseline_label"] = baseline_label
    report_data["test_label"] = test_label
    fname = f"{baseline_label}_vs_{test_label}.json"
    (REPORTS_DIR / fname).write_text(
        json.dumps(report_data, indent=2), encoding="utf-8")

    return report


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

def experiment_1(monitors):
    """Exp 1: Model Fingerprinting — run 25 prompts against each of 7 models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Model Fingerprinting")
    print("=" * 70)

    for name, cfg in MODELS.items():
        mon, data = run_condition(
            label=name,
            provider=cfg["provider"],
            model_id=cfg["model_id"],
            prompts=PROMPTS,
        )
        monitors[name] = mon
        print(f"  => {name}: {mon._count} responses, "
              f"score={mon.score}, verdict={mon.verdict}")


def experiment_2_3(monitors, results):
    """Exp 2-3: Intra-family + cross-provider detectability matrix."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2-3: Detectability Matrix (intra + cross-provider)")
    print("=" * 70)

    model_names = list(MODELS.keys())
    n = len(model_names)
    matrix_z = np.zeros((n, n))
    matrix_status = [["-"] * n for _ in range(n)]
    matrix_ndev = np.zeros((n, n), dtype=int)

    for i, bl_name in enumerate(model_names):
        for j, test_name in enumerate(model_names):
            if i == j:
                matrix_status[i][j] = "SELF"
                continue
            if test_name not in monitors:
                continue

            report = cross_check(bl_name, test_name, monitors[test_name])
            if report is None:
                continue

            max_wz = max(d["weighted_z"] for d in report.deviations)
            matrix_z[i, j] = max_wz
            matrix_status[i][j] = report.status
            matrix_ndev[i, j] = report.n_deviated

            provider_bl = MODELS[bl_name]["provider"]
            provider_test = MODELS[test_name]["provider"]
            cross = "CROSS" if provider_bl != provider_test else "INTRA"

            print(f"  {bl_name:16s} -> {test_name:16s}: "
                  f"{report.status:12s} max_wz={max_wz:6.2f} "
                  f"n_dev={report.n_deviated:2d} [{cross}]")

    results["matrix_z"] = matrix_z.tolist()
    results["matrix_status"] = matrix_status
    results["matrix_ndev"] = matrix_ndev.tolist()
    results["model_names"] = model_names

    return matrix_z, matrix_status, matrix_ndev


def experiment_4(monitors, results):
    """Exp 4: System Prompt Tampering — 3 styles x 3 models."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: System Prompt Tampering")
    print("=" * 70)

    tampering_results = []

    for model_name in TAMPER_MODELS:
        cfg = MODELS[model_name]
        for style_name, sys_prompt in TAMPER_STYLES.items():
            label = f"tamper_{model_name}_{style_name}"
            mon, data = run_condition(
                label=label,
                provider=cfg["provider"],
                model_id=cfg["model_id"],
                prompts=PROMPTS,
                system_prompt=sys_prompt,
            )
            monitors[label] = mon

            # Compare against normal baseline
            report = cross_check(model_name, label, mon)
            if report:
                max_wz = max(d["weighted_z"] for d in report.deviations)
                top_devs = sorted(
                    [d for d in report.deviations if d["deviated"]],
                    key=lambda d: d["weighted_z"], reverse=True,
                )[:3]
                top_str = ", ".join(
                    f"{d['name']}(z={d['weighted_z']:.1f})" for d in top_devs
                )

                result = {
                    "model": model_name,
                    "style": style_name,
                    "status": report.status,
                    "max_weighted_z": max_wz,
                    "n_deviated": report.n_deviated,
                    "top_deviations": top_str,
                }
                tampering_results.append(result)
                print(f"  {model_name:16s} + {style_name:8s}: "
                      f"{report.status:12s} max_wz={max_wz:6.2f} "
                      f"n_dev={report.n_deviated:2d}  [{top_str}]")

    results["tampering"] = tampering_results

    # Save summary
    (DATA_DIR / "tampering_results.json").write_text(
        json.dumps(tampering_results, indent=2), encoding="utf-8")


def experiment_5(monitors, results):
    """Exp 5: Temperature Drift — 3 temperatures on gpt-4o-mini."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Temperature Drift")
    print("=" * 70)

    cfg = MODELS[TEMP_MODEL]
    temp_results = []

    # Run 0.7 first (baseline), then the rest
    ordered_temps = sorted(TEMPERATURES, key=lambda t: (t != 0.7, t))

    for temp in ordered_temps:
        label = f"temp_{TEMP_MODEL}_{temp}"
        mon, data = run_condition(
            label=label,
            provider=cfg["provider"],
            model_id=cfg["model_id"],
            prompts=PROMPTS,
            temperature=temp,
        )
        monitors[label] = mon

        if temp == 0.7:
            print(f"  temp={temp}: baseline (reference)")
            continue

        # Compare against default temperature (0.7) baseline
        baseline_label = f"temp_{TEMP_MODEL}_0.7"
        if (BASELINES_DIR / f"{baseline_label}.json").exists():
            report = cross_check(baseline_label, label, mon)
            if report:
                max_wz = max(d["weighted_z"] for d in report.deviations)
                result = {
                    "temperature": temp,
                    "baseline_temp": 0.7,
                    "status": report.status,
                    "max_weighted_z": max_wz,
                    "n_deviated": report.n_deviated,
                }
                temp_results.append(result)
                print(f"  temp={temp} vs 0.7: {report.status:12s} "
                      f"max_wz={max_wz:.2f} n_dev={report.n_deviated}")

    results["temperature"] = temp_results

    (DATA_DIR / "temperature_results.json").write_text(
        json.dumps(temp_results, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════

def generate_figures(monitors, results):
    """Generate all paper figures from collected data."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    model_names = list(MODELS.keys())
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

    # ── Fig 1: Fingerprint Heatmap ─────────────────────────────────
    print("  Fig 1: Fingerprint heatmap...")
    fig, ax = plt.subplots(figsize=(14, 5))

    # Collect feature means per model
    feat_matrix = []
    for name in model_names:
        if name in monitors:
            fh = np.array(monitors[name].analyzer.feature_history)
            feat_matrix.append(fh.mean(axis=0))
        else:
            feat_matrix.append(np.zeros(N_FEATURES))
    feat_matrix = np.array(feat_matrix)

    # Z-score normalize per feature (column-wise)
    col_mean = feat_matrix.mean(axis=0)
    col_std = feat_matrix.std(axis=0) + 1e-10
    z_matrix = (feat_matrix - col_mean) / col_std

    cmap = LinearSegmentedColormap.from_list("ekg", ["#1a1a2e", "#0f3460", "#16213e",
                                                      "#e94560", "#ff6b6b"])
    im = ax.imshow(z_matrix, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    plt.colorbar(im, ax=ax, label="Z-score (normalized across models)")
    ax.set_title("Model Behavioral Fingerprints (z-score normalized)")

    # Annotate cells with values
    for i in range(len(model_names)):
        for j in range(N_FEATURES):
            val = z_matrix[i, j]
            color = "white" if abs(val) > 1.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=5, color=color)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_fingerprint_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Detectability Matrix ────────────────────────────────
    print("  Fig 2: Detectability matrix...")
    matrix_z = np.array(results.get("matrix_z", []))
    matrix_status = results.get("matrix_status", [])

    if matrix_z.size > 0:
        n = len(model_names)
        fig, ax = plt.subplots(figsize=(9, 8))

        # Color by status: CLEAN=green, WARNING=yellow, COMPROMISED=red, SELF=grey
        status_colors = np.zeros((n, n, 3))
        for i in range(n):
            for j in range(n):
                s = matrix_status[i][j]
                if s == "SELF":
                    status_colors[i, j] = [0.85, 0.85, 0.85]
                elif s == "CLEAN":
                    status_colors[i, j] = [0.2, 0.7, 0.3]
                elif s == "WARNING":
                    status_colors[i, j] = [0.95, 0.8, 0.2]
                elif s == "COMPROMISED":
                    status_colors[i, j] = [0.9, 0.2, 0.2]
                else:
                    status_colors[i, j] = [0.9, 0.9, 0.9]

        ax.imshow(status_colors, aspect="equal")
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "SELF", ha="center", va="center",
                            fontsize=7, color="gray")
                else:
                    wz = matrix_z[i, j]
                    st = matrix_status[i][j]
                    color = "white" if st == "COMPROMISED" else "black"
                    ax.text(j, i, f"{st}\nz={wz:.1f}", ha="center",
                            va="center", fontsize=6, color=color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(model_names, fontsize=8)
        ax.set_xlabel("Test Model")
        ax.set_ylabel("Baseline Model")
        ax.set_title("Cross-Model Detectability Matrix (3-sigma)")

        # Draw provider boundaries dynamically
        providers = [MODELS[mn]["provider"] for mn in model_names]
        boundaries = []
        for k in range(1, len(providers)):
            if providers[k] != providers[k - 1]:
                boundaries.append(k - 0.5)
        for pos in boundaries:
            ax.axhline(pos, color="white", linewidth=2)
            ax.axvline(pos, color="white", linewidth=2)

        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig2_detectability_matrix.png", bbox_inches="tight")
        plt.close(fig)

    # ── Fig 3: Tampering Radar Charts ──────────────────────────────
    print("  Fig 3: Tampering radar charts...")
    tamper_data = results.get("tampering", [])

    if tamper_data:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15),
                                 subplot_kw={"projection": "polar"})

        # Select top 8 features for readability
        top_features = ["length_words", "length_chars", "vocab_diversity",
                        "sentence_count", "avg_sentence_length", "hedge_ratio",
                        "list_ratio", "assertion_density"]
        feat_idx = [FEATURE_NAMES.index(f) for f in top_features]
        angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        for row, model_name in enumerate(TAMPER_MODELS):
            # Normal baseline
            if model_name in monitors:
                fh_normal = np.array(monitors[model_name].analyzer.feature_history)
                means_normal = fh_normal[:, feat_idx].mean(axis=0)
            else:
                means_normal = np.zeros(len(feat_idx))

            for col, (style_name, _) in enumerate(TAMPER_STYLES.items()):
                ax = axes[row, col]
                label_t = f"tamper_{model_name}_{style_name}"

                if label_t in monitors:
                    fh_tamper = np.array(monitors[label_t].analyzer.feature_history)
                    means_tamper = fh_tamper[:, feat_idx].mean(axis=0)
                else:
                    means_tamper = np.zeros(len(feat_idx))

                # Normalize to [0,1] range for each feature
                all_vals = np.stack([means_normal, means_tamper])
                vmin = all_vals.min(axis=0)
                vmax = all_vals.max(axis=0) + 1e-10
                norm_normal = (means_normal - vmin) / (vmax - vmin)
                norm_tamper = (means_tamper - vmin) / (vmax - vmin)

                vals_n = np.concatenate([norm_normal, [norm_normal[0]]])
                vals_t = np.concatenate([norm_tamper, [norm_tamper[0]]])

                ax.plot(angles, vals_n, "o-", linewidth=1.5, color="#2196F3",
                        label="Normal", markersize=3)
                ax.fill(angles, vals_n, alpha=0.15, color="#2196F3")
                ax.plot(angles, vals_t, "o-", linewidth=1.5, color="#f44336",
                        label=f"Tampered ({style_name})", markersize=3)
                ax.fill(angles, vals_t, alpha=0.15, color="#f44336")

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(top_features, fontsize=5)
                ax.set_title(f"{model_name}\n+ {style_name}", fontsize=8, pad=12)
                ax.legend(fontsize=5, loc="upper right")

        fig.suptitle("Normal vs Tampered Feature Profiles", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_tampering_radar.png", bbox_inches="tight")
        plt.close(fig)

    # ── Fig 4: Tampering Z-scores ──────────────────────────────────
    print("  Fig 4: Tampering z-scores...")
    if tamper_data:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

        for col, style_name in enumerate(TAMPER_STYLES.keys()):
            ax = axes[col]
            style_results = []

            for model_name in TAMPER_MODELS:
                label_t = f"tamper_{model_name}_{style_name}"
                report_file = REPORTS_DIR / f"{model_name}_vs_{label_t}.json"
                if report_file.exists():
                    rdata = json.loads(report_file.read_text())
                    for d in rdata["deviations"]:
                        style_results.append({
                            "model": model_name,
                            "feature": d["name"],
                            "weighted_z": d["weighted_z"],
                            "deviated": d["deviated"],
                        })

            # Group by feature, show max z across models
            feature_maxz = {}
            for r in style_results:
                f = r["feature"]
                if f not in feature_maxz or r["weighted_z"] > feature_maxz[f]:
                    feature_maxz[f] = r["weighted_z"]

            # Sort by max z
            sorted_feats = sorted(feature_maxz.items(), key=lambda x: x[1],
                                  reverse=True)
            feat_names = [f[0] for f in sorted_feats]
            z_vals = [f[1] for f in sorted_feats]
            colors = ["#f44336" if z > 3.0 else "#FFC107" if z > 1.5
                       else "#4CAF50" for z in z_vals]

            ax.barh(range(len(feat_names)), z_vals, color=colors)
            ax.set_yticks(range(len(feat_names)))
            ax.set_yticklabels(feat_names, fontsize=7)
            ax.axvline(3.0, color="red", linestyle="--", linewidth=1,
                       label="3-sigma threshold")
            ax.set_xlabel("Max Weighted Z-score")
            ax.set_title(f"Tampering: {style_name}", fontsize=10)
            ax.legend(fontsize=7)
            ax.invert_yaxis()

        fig.suptitle("Z-score Deviations by Tampering Style (max across models)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig4_tampering_zscores.png", bbox_inches="tight")
        plt.close(fig)

    # ── Fig 5: Temperature Drift ───────────────────────────────────
    print("  Fig 5: Temperature drift...")
    temp_results = results.get("temperature", [])

    if temp_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Feature comparison across temperatures
        temp_labels_all = [f"temp_{TEMP_MODEL}_{t}" for t in TEMPERATURES]
        temp_feat_means = []
        for tl in temp_labels_all:
            if tl in monitors:
                fh = np.array(monitors[tl].analyzer.feature_history)
                temp_feat_means.append(fh.mean(axis=0))
            else:
                temp_feat_means.append(np.zeros(N_FEATURES))
        temp_feat_means = np.array(temp_feat_means)

        # Top 6 most variable features
        feat_var = temp_feat_means.std(axis=0)
        top_idx = np.argsort(feat_var)[-6:][::-1]
        x = np.arange(len(top_idx))
        width = 0.25

        for i, temp in enumerate(TEMPERATURES):
            vals = temp_feat_means[i, top_idx]
            ax1.bar(x + i * width, vals, width, label=f"T={temp}",
                    color=["#2196F3", "#4CAF50", "#f44336"][i])

        ax1.set_xticks(x + width)
        ax1.set_xticklabels([FEATURE_NAMES[i] for i in top_idx],
                            rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("Feature Mean")
        ax1.set_title("Top Features Varying with Temperature")
        ax1.legend()

        # Z-score summary
        temps_tested = [r["temperature"] for r in temp_results]
        z_vals = [r["max_weighted_z"] for r in temp_results]
        statuses = [r["status"] for r in temp_results]
        colors = ["#f44336" if s == "COMPROMISED" else "#FFC107" if s == "WARNING"
                  else "#4CAF50" for s in statuses]
        bars = ax2.bar(range(len(temps_tested)),
                       z_vals, color=colors)
        ax2.set_xticks(range(len(temps_tested)))
        ax2.set_xticklabels([f"T={t}" for t in temps_tested])
        ax2.axhline(3.0, color="red", linestyle="--", label="3-sigma")
        ax2.set_ylabel("Max Weighted Z-score")
        ax2.set_title(f"Temperature Drift Detection ({TEMP_MODEL})")
        ax2.legend()

        # Annotate bars with status
        for bar, status in zip(bars, statuses):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     status, ha="center", fontsize=8, fontweight="bold")

        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig5_temperature_drift.png", bbox_inches="tight")
        plt.close(fig)

    # ── Fig 6: PCA Clusters ────────────────────────────────────────
    print("  Fig 6: PCA clusters...")
    all_features = []
    all_labels = []
    color_map = {}
    colors_list = ["#2196F3", "#f44336", "#4CAF50", "#FF9800",
                   "#9C27B0", "#00BCD4", "#795548", "#E91E63", "#607D8B"]

    for i, name in enumerate(model_names):
        if name in monitors:
            fh = np.array(monitors[name].analyzer.feature_history)
            all_features.append(fh)
            all_labels.extend([name] * fh.shape[0])
            color_map[name] = colors_list[i % len(colors_list)]

    if all_features:
        X = np.vstack(all_features)
        # Z-score normalize
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        # PCA via SVD
        U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
        pcs = X_norm @ Vt[:2].T
        var_explained = S[:2] ** 2 / (S ** 2).sum() * 100

        fig, ax = plt.subplots(figsize=(10, 8))
        for name in model_names:
            if name in color_map:
                mask = [l == name for l in all_labels]
                ax.scatter(pcs[mask, 0], pcs[mask, 1], c=color_map[name],
                           label=name, alpha=0.6, s=30, edgecolors="white",
                           linewidths=0.3)
                # Centroid
                cx = pcs[mask, 0].mean()
                cy = pcs[mask, 1].mean()
                ax.scatter(cx, cy, c=color_map[name], s=150, marker="X",
                           edgecolors="black", linewidths=1.5)

        ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
        ax.set_title("PCA of Response Feature Vectors by Model")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig6_pca_clusters.png", bbox_inches="tight")
        plt.close(fig)

    print("  All figures saved to", FIG_DIR)


# ══════════════════════════════════════════════════════════════════════
# LATEX TABLES
# ══════════════════════════════════════════════════════════════════════

def generate_tables(monitors, results):
    """Generate LaTeX tables for the paper."""
    print("\n" + "=" * 70)
    print("GENERATING LATEX TABLES")
    print("=" * 70)

    model_names = list(MODELS.keys())

    # ── Table 1: Feature Means per Model ───────────────────────────
    print("  Table 1: Feature means...")
    # Select 8 most discriminative features
    feat_matrix = []
    for name in model_names:
        if name in monitors:
            fh = np.array(monitors[name].analyzer.feature_history)
            feat_matrix.append(fh.mean(axis=0))
    feat_matrix = np.array(feat_matrix)

    # Most discriminative = highest CV across models
    cv = feat_matrix.std(axis=0) / (feat_matrix.mean(axis=0) + 1e-10)
    top8 = np.argsort(cv)[-8:][::-1]

    lines = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Mean feature values per model (top 8 most discriminative features).}",
        r"\label{tab:fingerprints}",
        r"\begin{tabular}{l" + "r" * 8 + "}",
        r"\toprule",
        r"\textbf{Model} & " + " & ".join(
            r"\texttt{" + FEATURE_NAMES[i].replace("_", r"\_") + "}"
            for i in top8
        ) + r" \\",
        r"\midrule",
    ]
    for mi, name in enumerate(model_names):
        vals = " & ".join(f"{feat_matrix[mi, i]:.2f}" for i in top8)
        lines.append(f"{name.replace('_', '-')} & {vals} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    (TABLE_DIR / "table1_fingerprints.tex").write_text("\n".join(lines))

    # ── Table 2: Detectability Matrix ──────────────────────────────
    print("  Table 2: Detectability matrix...")
    matrix_z = np.array(results.get("matrix_z", []))
    matrix_status = results.get("matrix_status", [])

    if matrix_z.size > 0:
        n = len(model_names)
        lines = [
            r"\begin{table*}[t]",
            r"\centering\small",
            r"\caption{Cross-model detectability matrix. Each cell shows the "
            r"security status when the column model is tested against the row "
            r"model's baseline ($\sigma=3$). Max weighted z-score in parentheses.}",
            r"\label{tab:detectability}",
            r"\begin{tabular}{l" + "c" * n + "}",
            r"\toprule",
            r"Baseline $\downarrow$ / Test $\rightarrow$ & " + " & ".join(
                r"\rotatebox{60}{" + mn.replace("_", "-") + "}"
                for mn in model_names
            ) + r" \\",
            r"\midrule",
        ]
        for i, bl in enumerate(model_names):
            cells = []
            for j, ts in enumerate(model_names):
                if i == j:
                    cells.append(r"\cellcolor{gray!20}---")
                else:
                    s = matrix_status[i][j]
                    z = matrix_z[i, j]
                    if s == "COMPROMISED":
                        cells.append(r"\cellcolor{red!25}" + f"C({z:.1f})")
                    elif s == "WARNING":
                        cells.append(r"\cellcolor{yellow!25}" + f"W({z:.1f})")
                    else:
                        cells.append(r"\cellcolor{green!15}" + f"OK({z:.1f})")
            lines.append(bl.replace("_", "-") + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

        (TABLE_DIR / "table2_detectability.tex").write_text("\n".join(lines))

    # ── Table 3: Tampering Results ─────────────────────────────────
    print("  Table 3: Tampering results...")
    tamper_data = results.get("tampering", [])

    if tamper_data:
        lines = [
            r"\begin{table}[h]",
            r"\centering\small",
            r"\caption{System prompt tampering detection results. All tampering "
            r"styles tested against the normal baseline of each model ($\sigma=3$).}",
            r"\label{tab:tampering}",
            r"\begin{tabular}{llcrrl}",
            r"\toprule",
            r"\textbf{Model} & \textbf{Style} & \textbf{Status} & "
            r"\textbf{Max $w_z$} & \textbf{\#Dev} & \textbf{Top Deviations} \\",
            r"\midrule",
        ]
        for r in tamper_data:
            status_tex = r["status"]
            if status_tex == "COMPROMISED":
                status_tex = r"\textcolor{red}{COMP.}"
            elif status_tex == "WARNING":
                status_tex = r"\textcolor{orange}{WARN}"
            else:
                status_tex = r"\textcolor{green!60!black}{CLEAN}"

            # Shorten top deviations for table
            top = r["top_deviations"][:40]
            top = top.replace("_", r"\_")

            lines.append(
                f"{r['model'].replace('_', '-')} & {r['style']} & "
                f"{status_tex} & {r['max_weighted_z']:.1f} & "
                f"{r['n_deviated']} & \\scriptsize{{{top}}} " + r"\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        (TABLE_DIR / "table3_tampering.tex").write_text("\n".join(lines))

    # ── Table 4: Temperature Results ───────────────────────────────
    print("  Table 4: Temperature results...")
    temp_data = results.get("temperature", [])

    if temp_data:
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Temperature drift detection on " +
            TEMP_MODEL.replace("_", "-") +
            r". Baseline: T=0.7 (default).}",
            r"\label{tab:temperature}",
            r"\begin{tabular}{cccc}",
            r"\toprule",
            r"\textbf{Temperature} & \textbf{Status} & "
            r"\textbf{Max $w_z$} & \textbf{\#Deviated} \\",
            r"\midrule",
        ]
        for r in temp_data:
            lines.append(
                f"{r['temperature']} & {r['status']} & "
                f"{r['max_weighted_z']:.2f} & {r['n_deviated']} " + r"\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        (TABLE_DIR / "table4_temperature.tex").write_text("\n".join(lines))

    print("  All tables saved to", TABLE_DIR)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("LLM EKG — Cross-Provider Behavioral Audit")
    print("=" * 70)

    # Verify API keys
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not XAI_API_KEY:
        missing.append("XAI_API_KEY")
    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if missing:
        print(f"WARNING: Missing API keys: {', '.join(missing)}")
        print("Set them as environment variables to run all experiments.")
        print("Continuing with available keys...\n")

    monitors = {}  # label -> LiveMonitor
    results = {}   # aggregate results for figures/tables

    t_start = time.time()

    # Run experiments
    experiment_1(monitors)
    experiment_2_3(monitors, results)
    experiment_4(monitors, results)
    experiment_5(monitors, results)

    elapsed_total = time.time() - t_start
    print(f"\n  Total experiment time: {elapsed_total:.0f}s "
          f"({elapsed_total / 60:.1f} min)")

    # Generate outputs
    generate_figures(monitors, results)
    generate_tables(monitors, results)

    # Save master results
    (DATA_DIR / "results_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print(f"  Data:    {DATA_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TABLE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
