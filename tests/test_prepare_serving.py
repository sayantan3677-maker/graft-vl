import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_hybrid_serving as p  # noqa: E402

PLAIN = "{%- for message in messages %}\n    {%- if message.content is string %}{{ message.content }}{%- endif %}\n{%- endfor %}"


def test_is_merged_detects_prepared_templates():
    assert not p.is_merged(PLAIN)
    assert p.is_merged(p.HEADER + PLAIN)


def test_original_template_skips_merged_copies(tmp_path):
    # an export initialised from a prepared export: chat_template.jinja is already merged, the saved original is genuine
    (tmp_path / "chat_template.jinja").write_text(p.HEADER + "merged")
    (tmp_path / "chat_template.qwen3_text_original.jinja").write_text(PLAIN)
    assert p.original_template(tmp_path, base="unused") == PLAIN


def test_original_template_never_saves_a_merged_original(tmp_path):
    # first prep of such an export: no saved original yet, but the plain template is the current one
    (tmp_path / "chat_template.jinja").write_text(PLAIN)
    assert p.original_template(tmp_path, base="unused") == PLAIN
    assert (tmp_path / "chat_template.qwen3_text_original.jinja").read_text() == PLAIN
