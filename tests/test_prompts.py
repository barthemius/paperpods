from paperpods.pdf_extract import ExtractedPaper
from paperpods.prompts import build_script_messages


def test_prompt_includes_formula_snippets_and_policy() -> None:
    paper = ExtractedPaper(
        title="A Theory of Everything",
        text="This is the extracted paper content.",
        formula_snippets=["E = mc^2", "argmax_x f(x)"],
    )
    system_prompt, user_prompt = build_script_messages(
        paper,
        title_override=None,
        style="single-narrator-15-20min",
        formula_policy="conceptual-plus-key-variables",
    )

    assert "expert scientific editor" in system_prompt
    assert "E = mc^2" in user_prompt
    assert "argmax_x f(x)" in user_prompt
    assert "define key variables in words" in user_prompt
    assert "text-to-speech reader" in user_prompt
    assert "spoken English" in user_prompt
    assert "raw mathematical notation" in user_prompt
    assert "the integral from zero to infinity" in user_prompt

