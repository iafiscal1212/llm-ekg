#!/usr/bin/env python3
"""
demo.py — Synthetic degradation demo for LLM EKG.

Generates 100 responses in 3 phases:
  1. Normal (50): long, diverse, low repetition
  2. Degradation (30): shorter, more hedging, more repetition
  3. Critical (20): hallucination + evasion patterns

Outputs demo_ekg_report.html.

(c) 2026 Carmen Esteban — IAFISCAL & PARTNERS
"""

import random
from pathlib import Path
from llm_ekg import LLMAnalyzer
from llm_ekg.report import EKGReport

_TECH_WORDS = [
    "algorithm", "neural", "network", "transformer", "attention", "gradient",
    "backpropagation", "optimization", "inference", "model", "training",
    "dataset", "feature", "embedding", "tokenizer", "encoder", "decoder",
    "architecture", "convolutional", "recurrent", "reinforcement",
    "supervised", "unsupervised", "classification", "regression",
    "precision", "recall", "accuracy", "benchmark", "evaluation",
    "deployment", "scalability", "distributed", "parallel", "computation",
    "tensor", "matrix", "vector", "dimension", "hyperparameter",
    "regularization", "dropout", "batch", "epoch", "learning",
    "momentum", "convergence", "loss", "objective", "function",
    "activation", "sigmoid", "softmax", "normalization", "layer",
    "pooling", "convolution", "stride", "kernel", "filter",
    "generative", "discriminative", "adversarial", "variational",
    "autoencoder", "diffusion", "sampling", "probability", "distribution",
    "entropy", "information", "mutual", "divergence", "likelihood",
]

_HEDGE_PHRASES = [
    "However, it's worth noting that",
    "Perhaps more importantly,",
    "It could be argued that",
    "Generally speaking,",
    "Although it might seem that",
    "Approximately speaking,",
    "Presumably,",
    "Relatively speaking,",
    "Arguably,",
    "Possibly,",
    "Somewhat related to this,",
    "Typically,",
]

_FILLER_SENTENCES = [
    "This is an important consideration in the field.",
    "The implications of this approach are significant.",
    "Further research is needed to fully understand the implications.",
    "The trade-offs should be carefully evaluated.",
    "This builds on previous work in the area.",
    "The methodology follows established best practices.",
    "Results may vary depending on the specific use case.",
    "Implementation details are crucial for performance.",
    "The theoretical foundations are well-established.",
    "Practical applications continue to evolve rapidly.",
]


def _generate_normal(rng, idx):
    n_paragraphs = rng.randint(3, 6)
    paragraphs = []
    for p in range(n_paragraphs):
        n_sentences = rng.randint(3, 7)
        sentences = []
        for s in range(n_sentences):
            n_words = rng.randint(10, 25)
            words = rng.sample(_TECH_WORDS, min(n_words, len(_TECH_WORDS)))
            if s > 0 and rng.random() < 0.3:
                conn = rng.choice(["Furthermore, ", "Additionally, ", "Moreover, ",
                                    "In particular, ", "Specifically, "])
                sentence = conn + " ".join(words) + "."
            else:
                sentence = " ".join(words).capitalize() + "."
            sentences.append(sentence)
        paragraphs.append(" ".join(sentences))
    if rng.random() < 0.4:
        items = []
        for _ in range(rng.randint(3, 6)):
            words = rng.sample(_TECH_WORDS, rng.randint(3, 8))
            items.append(f"- {' '.join(words).capitalize()}")
        paragraphs.insert(rng.randint(1, len(paragraphs)), "\n".join(items))
    if rng.random() < 0.3:
        code = "```python\ndef process(data):\n    result = transform(data)\n    return optimize(result)\n```"
        paragraphs.insert(rng.randint(1, len(paragraphs)), code)
    return "\n\n".join(paragraphs)


def _generate_degraded(rng, idx, severity):
    n_paragraphs = max(1, int(3 * (1 - severity * 0.6)))
    paragraphs = []
    for p in range(n_paragraphs):
        n_sentences = max(1, int(4 * (1 - severity * 0.5)))
        sentences = []
        for s in range(n_sentences):
            if rng.random() < 0.15 + severity * 0.4:
                hedge = rng.choice(_HEDGE_PHRASES)
                words = rng.sample(_TECH_WORDS, rng.randint(4, 10))
                sentence = f"{hedge} {' '.join(words)}."
            else:
                vocab_size = max(10, int(len(_TECH_WORDS) * (1 - severity * 0.6)))
                restricted = _TECH_WORDS[:vocab_size]
                n_words = max(5, int(15 * (1 - severity * 0.5)))
                words = [rng.choice(restricted) for _ in range(n_words)]
                sentence = " ".join(words).capitalize() + "."
            sentences.append(sentence)
        if rng.random() < severity * 0.5:
            sentences.append(rng.choice(_FILLER_SENTENCES))
        paragraphs.append(" ".join(sentences))
    if rng.random() < severity * 0.6:
        paragraphs.append(rng.choice(["I hope that helps.", "Let me know.", "I hope this helps."]))
    return "\n\n".join(paragraphs)


def _generate_critical(rng, idx):
    if idx % 2 == 0:
        # Type A: evasive
        templates = [
            "Perhaps I could help with that. However, it's worth noting that {w1} and {w2} are {w3}. I hope this helps.",
            "Generally speaking, {w1} might be related to {w2}. Although it might seem that {w3} is important, it could be argued that it depends.",
            "I think {w1} and {w2} are both relevant here. However, it could be argued that {w3} is also important. Relatively speaking, this is a complex topic.",
        ]
        template = rng.choice(templates)
        words = rng.sample(_TECH_WORDS, 3)
        return template.format(w1=words[0], w2=words[1], w3=words[2])
    else:
        # Type B: hallucination (fabricated details + absolute certainty)
        fake_numbers = [
            f"{rng.randint(10,99)}.{rng.randint(1,9)}%",
            f"{rng.randint(1000,9999)}",
            f"{rng.randint(2018,2025)}",
            f"${rng.randint(1,999)},{rng.randint(100,999)}",
        ]
        halluc_templates = [
            "The {w1} architecture was published on {d1} and achieved exactly {n1} accuracy on the benchmark. "
            "This is definitively the best approach. The authors confirmed {n2} improvements across all metrics. "
            "Every researcher agrees this is the correct method.",
            "According to the {d1} study by Dr. Smith et al., {w1} always outperforms {w2}. "
            "The exact improvement is {n1}, measured precisely across {n2} datasets. "
            "This has never been disputed. The results are absolutely clear.",
            "The {w1} method was developed in {d1} specifically for {w2}. It always produces exactly {n1} "
            "improvement. The {w3} variant never works. This is precisely documented in {n2} papers.",
        ]
        template = rng.choice(halluc_templates)
        words = rng.sample(_TECH_WORDS, 3)
        nums = rng.sample(fake_numbers, 2)
        return template.format(
            w1=words[0], w2=words[1], w3=words[2],
            n1=nums[0], n2=nums[1],
            d1=f"{rng.randint(1,28)}/{rng.randint(1,12)}/{rng.randint(2020,2025)}",
        )


def generate_synthetic_conversation(seed=42):
    rng = random.Random(seed)
    responses = []
    base_time = 1711670400.0  # 2024-03-29 00:00:00 UTC

    for i in range(100):
        ts = base_time + i * 300
        if i < 50:
            text = _generate_normal(rng, i)
            latency = rng.uniform(1.0, 4.0)
        elif i < 80:
            severity = (i - 50) / 30.0
            text = _generate_degraded(rng, i, severity)
            latency = rng.uniform(2.0, 8.0)
        else:
            text = _generate_critical(rng, i)
            latency = rng.uniform(5.0, 15.0)
        responses.append({
            "response": text,
            "timestamp": ts,
            "response_time_s": latency,
            "model": "synthetic-demo",
        })
    return responses


def main():
    print("LLM EKG — Synthetic Degradation Demo")
    print("=" * 50)
    print("\nGenerating 100 synthetic responses...")
    print("  Phase 1 (normal): responses 1-50")
    print("  Phase 2 (degradation): responses 51-80")
    print("  Phase 3 (critical): responses 81-100")

    responses = generate_synthetic_conversation(seed=42)

    print("\nAnalyzing...")
    analyzer = LLMAnalyzer(n_scales=6)

    for i, r in enumerate(responses):
        result = analyzer.ingest(
            response=r["response"],
            timestamp=r["timestamp"],
            response_time_s=r["response_time_s"],
        )
        if (i + 1) % 25 == 0:
            score = result["state"]["anomaly_score"]
            drift = result["state"]["drift"]
            print(f"  [{i+1:3d}/100] anomaly={score:.4f} drift={drift:.4f}")

    summary = analyzer.get_summary()
    print(f"\n{'=' * 50}")
    print(f"RESULT: {summary['verdict']} ({summary['global_score_100']}/100)")
    print(f"Trend: {summary['trend']:+.4f}")
    print(f"Hallucination risk: {summary['hallucination_risk']:.2%}")

    if summary.get("multiscale") and summary["multiscale"].get("mean_hurst"):
        print(f"Mean persistence: {summary['multiscale']['mean_hurst']:.3f}")

    output = str(Path(__file__).parent / "demo_ekg_report.html")
    report = EKGReport(analyzer)
    report.generate(output)
    print(f"\nReport: {output}")
    print("Open in browser to see the full EKG.")


if __name__ == "__main__":
    main()
