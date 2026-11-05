from __future__ import annotations

from .pdf_extract import ExtractedPaper

SCRIPT_STYLE_GUIDANCE = {
    "single-narrator-15-20min": "Write a single-narrator deep-dive script targeted at ~15-20 minutes of listening.",
    "single-narrator-8-12min": "Write a single-narrator concise expert script targeted at ~8-12 minutes of listening.",
    "two-host-dialogue-15-20min": "Write a two-host dialogue script targeted at ~15-20 minutes, with strong technical depth.",
}

FORMULA_POLICY_GUIDANCE = {
    "conceptual-plus-key-variables": (
        "Do not read formulas symbol-by-symbol unless unavoidable. "
        "Instead explain what each key equation means conceptually and define key variables in words."
    ),
    "high-level-only": "Summarize equation purpose at high level without variable-by-variable detail.",
    "detailed-derivation-summary": "Include compact derivation summaries where useful, while staying audio-friendly.",
}


def build_script_messages(
    paper: ExtractedPaper,
    *,
    title_override: str | None,
    style: str,
    formula_policy: str,
) -> tuple[str, str]:
    style_text = SCRIPT_STYLE_GUIDANCE.get(style, SCRIPT_STYLE_GUIDANCE["single-narrator-15-20min"])
    formula_text = FORMULA_POLICY_GUIDANCE.get(
        formula_policy,
        FORMULA_POLICY_GUIDANCE["conceptual-plus-key-variables"],
    )
    paper_title = title_override or paper.title
    formula_examples = "\n".join(f"- {line}" for line in paper.formula_snippets[:10]) or "- (none detected)"

    system_prompt = (
        "You are an expert scientific editor writing podcast scripts for domain experts. "
        "Preserve technical fidelity while making spoken narration coherent."
    )
    user_prompt = f"""
Generate a podcast script from the scientific paper content below.

This script is NOT for a human reader taking notes. It is optimized for a text-to-speech reader that will speak the final text aloud.

Constraints:
- Audience: researchers and advanced practitioners.
- Keep technical precision; do not over-simplify.
- {style_text}
- {formula_text}
- Avoid hallucinating results not in the source.
- Include: motivation, method, core equations explained in words, main results, limitations, and practical implications.
- Use plain spoken prose suitable for text-to-speech.
- Keep the final script under 10,000 characters total.
- Every symbol, equation, integral, sum, limit, derivative, matrix, fraction, vector, tensor and special function must be written in spoken English, never as raw mathematical notation.
- Replace raw mathematical notation with explicit verbal descriptions such as:
  - "the integral from zero to infinity"
  - "the energy equals mass times the speed of light squared"
  - "the sum over i"
  - "the partial derivative with respect to time"
  - "the gradient"
  - "the value that maximizes the objective"
- When a paper contains equations, describe the meaning and the role of each important term in ordinary readable language.
- Do not include raw LaTeX, code-like math snippets, or formula notation in the final script.
- Avoid markdown tables and code fences.

Paper title: {paper_title}

Potential formula-heavy snippets:
{formula_examples}

Paper content:
{paper.text}
""".strip()
    return system_prompt, user_prompt

