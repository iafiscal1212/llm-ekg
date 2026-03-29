"""
CLI entry point and input parsers for LLM EKG.

Usage:
    python -m llm_ekg conversation.json --output report.html
    python -m llm_ekg --format chatgpt export.json
    python -m llm_ekg --format text responses.txt
    python -m llm_ekg --format api_log log.csv
    python -m llm_ekg --live openai  # live API monitoring

(c) 2026 Carmen Esteban — IAFISCAL & PARTNERS
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .engine import LLMAnalyzer, FEATURE_NAMES


# ── Parsers ──────────────────────────────────────────────────────────────

def parse_chatgpt_export(filepath: str) -> list[dict]:
    """Parse ChatGPT JSON export (conversations.json)."""
    data = json.loads(Path(filepath).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    responses = []
    for conv in data:
        mapping = conv.get("mapping", {})
        for node_id, node in mapping.items():
            msg = node.get("message")
            if msg is None:
                continue
            if msg.get("author", {}).get("role") != "assistant":
                continue
            content = msg.get("content", {})
            parts = content.get("parts", [])
            text = ""
            for p in parts:
                if isinstance(p, str):
                    text += p
            if not text.strip():
                continue
            ts = msg.get("create_time", 0.0) or 0.0
            responses.append({
                "response": text,
                "timestamp": float(ts),
                "model": msg.get("metadata", {}).get("model_slug", ""),
            })

    responses.sort(key=lambda r: r["timestamp"])
    return responses


def parse_claude_export(filepath: str) -> list[dict]:
    """Parse Claude JSON export (single file or directory)."""
    p = Path(filepath)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]

    responses = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        messages = []
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict):
            messages = data.get("chat_messages", data.get("messages", []))

        for msg in messages:
            if msg.get("sender") == "human" or msg.get("role") == "user":
                continue
            text = msg.get("text", msg.get("content", ""))
            if isinstance(text, list):
                text = " ".join(
                    b.get("text", "") for b in text
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not text or not text.strip():
                continue
            ts = msg.get("created_at", msg.get("timestamp", 0.0))
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                except (ValueError, TypeError):
                    ts = 0.0
            responses.append({
                "response": text,
                "timestamp": float(ts) if ts else 0.0,
                "model": msg.get("model", ""),
            })

    responses.sort(key=lambda r: r["timestamp"])
    return responses


def parse_api_log(filepath: str) -> list[dict]:
    """Parse CSV API log: timestamp,model,prompt,response,latency_ms."""
    responses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("response", "")
            if not text.strip():
                continue
            ts = 0.0
            if "timestamp" in row and row["timestamp"]:
                try:
                    ts = float(row["timestamp"])
                except ValueError:
                    try:
                        ts = datetime.fromisoformat(
                            row["timestamp"].replace("Z", "+00:00")).timestamp()
                    except (ValueError, TypeError):
                        ts = 0.0
            latency = 0.0
            if "latency_ms" in row and row["latency_ms"]:
                try:
                    latency = float(row["latency_ms"]) / 1000.0
                except ValueError:
                    pass
            responses.append({
                "response": text,
                "timestamp": ts,
                "response_time_s": latency,
                "model": row.get("model", ""),
            })
    return responses


def parse_generic_text(filepath: str) -> list[dict]:
    """Parse plain text: each block separated by blank line = one response."""
    text = Path(filepath).read_text(encoding="utf-8")
    blocks = re.split(r'\n\s*\n', text)
    responses = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        responses.append({
            "response": block,
            "timestamp": float(i),
        })
    return responses


def parse_jsonl(filepath: str) -> list[dict]:
    """Parse JSONL file (one JSON object per line)."""
    responses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("response", obj.get("content", obj.get("text", "")))
            if not text or not text.strip():
                continue
            ts = obj.get("timestamp", obj.get("created_at", 0.0))
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                except (ValueError, TypeError):
                    ts = 0.0
            latency = obj.get("response_time_s", obj.get("latency_ms", 0.0))
            if "latency_ms" in obj and latency > 100:  # was in ms
                latency = latency / 1000.0
            responses.append({
                "response": text,
                "timestamp": float(ts) if ts else 0.0,
                "response_time_s": float(latency) if latency else 0.0,
                "model": obj.get("model", ""),
            })
    return responses


def auto_detect_format(filepath: str) -> str:
    """Auto-detect input format from extension and content."""
    p = Path(filepath)
    if p.is_dir():
        return "claude"
    ext = p.suffix.lower()
    if ext == ".csv":
        return "api_log"
    if ext == ".txt":
        return "text"
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "mapping" in first:
                        return "chatgpt"
                    if "chat_messages" in first or "sender" in first:
                        return "claude"
            elif isinstance(data, dict):
                if "mapping" in data:
                    return "chatgpt"
                if "chat_messages" in data:
                    return "claude"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return "chatgpt"
    return "text"


def parse_input(filepath: str, fmt: str = "auto") -> list[dict]:
    """Parse input file according to format."""
    if fmt == "auto":
        fmt = auto_detect_format(filepath)

    parsers = {
        "chatgpt": parse_chatgpt_export,
        "claude": parse_claude_export,
        "api_log": parse_api_log,
        "text": parse_generic_text,
        "jsonl": parse_jsonl,
    }

    parser = parsers.get(fmt)
    if parser is None:
        raise ValueError(f"Unknown format: {fmt}. Use: {list(parsers.keys())}")
    return parser(filepath)


# ── Live API monitoring ──────────────────────────────────────────────────

class LiveMonitor:
    """Wraps OpenAI or Anthropic client to monitor responses in real-time.

    Usage:
        from llm_ekg import LiveMonitor

        monitor = LiveMonitor()
        # Wrap an existing client
        wrapped_openai = monitor.wrap_openai(openai_client)
        # Use wrapped_openai exactly like the original
        response = wrapped_openai.chat.completions.create(...)
        # Get report at any time
        monitor.report("my_report.html")
    """

    def __init__(self, **analyzer_kwargs):
        self.analyzer = LLMAnalyzer(**analyzer_kwargs)
        self._count = 0

    def ingest(self, response_text: str, response_time_s: float = 0.0,
               model: str = "", **extra):
        """Manually ingest a response."""
        import time
        result = self.analyzer.ingest(
            response=response_text,
            timestamp=time.time(),
            response_time_s=response_time_s,
            model=model, **extra,
        )
        self._count += 1
        return result

    def wrap_openai(self, client):
        """Wrap an OpenAI client to auto-monitor completions.

        Returns a proxy that intercepts chat.completions.create() calls.
        """
        monitor = self

        class _CompletionsProxy:
            def __init__(self, original):
                self._original = original

            def create(self, **kwargs):
                import time
                t0 = time.time()
                result = self._original.create(**kwargs)
                elapsed = time.time() - t0
                # Extract response text
                try:
                    text = result.choices[0].message.content or ""
                    model = result.model or kwargs.get("model", "")
                    monitor.ingest(text, response_time_s=elapsed, model=model)
                except (AttributeError, IndexError):
                    pass
                return result

        class _ChatProxy:
            def __init__(self, original):
                self.completions = _CompletionsProxy(original.completions)

        class _ClientProxy:
            def __init__(self, original):
                self._original = original
                self.chat = _ChatProxy(original.chat)

            def __getattr__(self, name):
                return getattr(self._original, name)

        return _ClientProxy(client)

    def wrap_anthropic(self, client):
        """Wrap an Anthropic client to auto-monitor messages.

        Returns a proxy that intercepts messages.create() calls.
        """
        monitor = self

        class _MessagesProxy:
            def __init__(self, original):
                self._original = original

            def create(self, **kwargs):
                import time
                t0 = time.time()
                result = self._original.create(**kwargs)
                elapsed = time.time() - t0
                try:
                    text = ""
                    for block in result.content:
                        if hasattr(block, 'text'):
                            text += block.text
                    model = result.model or kwargs.get("model", "")
                    monitor.ingest(text, response_time_s=elapsed, model=model)
                except (AttributeError, IndexError):
                    pass
                return result

        class _ClientProxy:
            def __init__(self, original):
                self._original = original
                self.messages = _MessagesProxy(original.messages)

            def __getattr__(self, name):
                return getattr(self._original, name)

        return _ClientProxy(client)

    def save_baseline(self, path: str, model: str = ""):
        """Save current session as a security baseline."""
        from .security import SecurityBaseline
        bl = SecurityBaseline.from_analyzer(self.analyzer, model=model)
        bl.save(path)
        return bl

    def security_check(self, baseline_path: str, sigma: float = 3.0):
        """Compare current session against a saved baseline."""
        from .security import SecurityBaseline
        bl = SecurityBaseline.load(baseline_path)
        return bl.check(self.analyzer, sigma=sigma)

    def summary(self) -> dict:
        return self.analyzer.get_summary()

    def report(self, output_path: str = "llm_ekg_report.html",
               security_report=None):
        from .report import EKGReport
        r = EKGReport(self.analyzer, security_report=security_report)
        r.generate(output_path)
        print(f"Report generated: {output_path} ({self._count} responses)")

    @property
    def score(self) -> int:
        s = self.analyzer.get_summary()
        return s.get("global_score_100", 0)

    @property
    def verdict(self) -> str:
        s = self.analyzer.get_summary()
        return s.get("verdict", "NO DATA")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM EKG — Mathematical Health Monitor for LLMs",
        epilog="Examples:\n"
               "  python -m llm_ekg conversations.json\n"
               "  python -m llm_ekg --format chatgpt export.json -o report.html\n"
               "  python -m llm_ekg --format text responses.txt\n"
               "  python -m llm_ekg --format api_log log.csv\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Conversation file (JSON/CSV/TXT/JSONL)")
    parser.add_argument(
        "--format", "-f",
        choices=["auto", "chatgpt", "claude", "api_log", "text", "jsonl"],
        default="auto",
        help="Input file format (default: auto-detect)",
    )
    parser.add_argument(
        "--output", "-o",
        default="llm_ekg_report.html",
        help="Output HTML report path (default: llm_ekg_report.html)",
    )
    parser.add_argument(
        "--n-scales", type=int, default=6,
        help="Number of analysis scales (default: 6)",
    )
    parser.add_argument(
        "--baseline",
        metavar="OUTPUT_PATH",
        help="Capture security baseline and save to JSON file",
    )
    parser.add_argument(
        "--security-check",
        metavar="BASELINE_PATH",
        help="Compare against a saved security baseline",
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0,
        help="Deviation threshold in sigmas for security check (default: 3.0)",
    )

    args = parser.parse_args()

    from . import __version__
    print(f"LLM EKG v{__version__}")
    print(f"Input: {args.input}")

    responses = parse_input(args.input, args.format)
    print(f"Responses found: {len(responses)}")

    if len(responses) < 5:
        print("ERROR: Need at least 5 responses for meaningful analysis.")
        sys.exit(1)

    analyzer = LLMAnalyzer(n_scales=args.n_scales)

    for i, r in enumerate(responses):
        result = analyzer.ingest(
            response=r.get("response", ""),
            timestamp=r.get("timestamp", 0.0),
            response_time_s=r.get("response_time_s", 0.0),
            model=r.get("model", ""),
        )
        if (i + 1) % 50 == 0 or i + 1 == len(responses):
            score = result["state"]["anomaly_score"]
            print(f"  [{i+1:4d}/{len(responses)}] anomaly={score:.4f}")

    summary = analyzer.get_summary()
    print(f"\n{'=' * 50}")
    print(f"RESULT: {summary['verdict']} ({summary['global_score_100']}/100)")
    print(f"Trend: {summary['trend']:+.4f}")
    print(f"Hallucination risk: {summary['hallucination_risk']:.2%}")

    if summary.get("multiscale") and summary["multiscale"].get("mean_hurst"):
        print(f"Mean persistence: {summary['multiscale']['mean_hurst']:.3f}")

    # Security: capture baseline
    sec_report = None
    if args.baseline:
        from .security import SecurityBaseline
        bl = SecurityBaseline.from_analyzer(analyzer)
        bl.save(args.baseline)
        print(f"\nBaseline saved: {args.baseline} ({summary['n_responses']} responses)")

    # Security: check against baseline
    if args.security_check:
        from .security import SecurityBaseline
        bl = SecurityBaseline.load(args.security_check)
        sec_report = bl.check(analyzer, sigma=args.sigma)
        status_color = {
            "CLEAN": "\033[32m",
            "WARNING": "\033[33m",
            "COMPROMISED": "\033[31m",
        }
        reset = "\033[0m"
        c = status_color.get(sec_report.status, "")
        print(f"\nSECURITY: {c}{sec_report.status}{reset} "
              f"({sec_report.n_deviated}/{len(sec_report.deviations)} features deviated, "
              f"threshold: {sec_report.sigma_threshold}\u03c3)")
        for d in sec_report.deviations:
            if d["deviated"]:
                print(f"  ! {d['name']}: baseline={d['baseline_mean']:.4f} "
                      f"current={d['current_mean']:.4f} z={d['weighted_z']:.2f}")

    from .report import EKGReport
    report = EKGReport(analyzer, security_report=sec_report)
    report.generate(args.output)
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
