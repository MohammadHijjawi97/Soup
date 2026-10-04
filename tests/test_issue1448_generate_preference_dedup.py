"""#1448 — ``soup data generate --dedup-with`` skipped ``--template preference`` rows.

A preference row fails the alpaca / sharegpt / chatml shape check, passes
``_validate_preference``, and was appended right there, before the dedup check.
So for DPO / ORPO and KTO rows ``--dedup-with`` was loaded and announced but
never applied, and a row the model repeated across batches was kept every time.

This file pins:

* a DPO / KTO row already in the ``--dedup-with`` file is dropped and counted;
* a row repeated across batches of one run is kept once, with no
  ``--dedup-with`` at all;
* preference rows are compared on their preference fields, so key order and
  extra keys in the file do not hide a repeat, and a KTO row with the opposite
  ``label`` (``false`` used to vanish from the key) is not a repeat;
* rows that differ are all kept, and the alpaca path is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from typer.testing import CliRunner

import soup_cli.commands.generate as generate
from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

DPO = {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
KTO_GOOD = {"prompt": "What is 2+2?", "completion": "4", "label": True}
KTO_BAD = {"prompt": "What is 2+2?", "completion": "4", "label": False}


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # generate refuses paths outside the cwd


def _model_returns(monkeypatch: pytest.MonkeyPatch, rows: List[dict]) -> None:
    """Stand in for the provider: batch ``i`` returns ``rows[i]``."""
    queue = [dict(row) for row in rows]

    def fake_generate_openai(**_kwargs):
        return [queue.pop(0)]

    monkeypatch.setattr(generate, "_generate_openai", fake_generate_openai)


def _write_prior(rows: List[dict]) -> None:
    Path("prior.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _generate(count: int, *extra: str):
    result = runner.invoke(
        app,
        [
            "data", "generate",
            "-p", "x",
            "-n", str(count),
            "--batch-size", "1",
            "--rpm", "100000",
            "-o", "out.jsonl",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    lines = Path("out.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()], strip_ansi(result.output)


@pytest.mark.parametrize(
    ("row", "extra"),
    [(DPO, []), (KTO_GOOD, ["--pref-task", "kto"]), (KTO_BAD, ["--pref-task", "kto"])],
    ids=["dpo", "kto-true", "kto-false"],
)
def test_dedup_with_drops_a_preference_row_already_in_the_file(
    monkeypatch: pytest.MonkeyPatch, row: dict, extra: List[str]
) -> None:
    _write_prior([row])
    _model_returns(monkeypatch, [row, row, row])

    kept, output = _generate(
        3, "--template", "preference", "--dedup-with", "prior.jsonl", *extra
    )

    assert kept == []
    assert "Duplicates: 3 removed" in output


def test_a_preference_row_repeated_across_batches_is_kept_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _model_returns(monkeypatch, [DPO, DPO, DPO])

    kept, output = _generate(3, "--template", "preference")

    assert kept == [DPO]
    assert "Duplicates: 2 removed" in output


def test_key_order_and_extra_keys_in_the_file_do_not_hide_a_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_prior([{"id": 7, "rejected": "5", "chosen": "4", "prompt": "What is 2+2?"}])
    _model_returns(monkeypatch, [DPO])

    kept, output = _generate(1, "--template", "preference", "--dedup-with", "prior.jsonl")

    assert kept == []
    assert "Duplicates: 1 removed" in output


def test_a_kto_row_with_the_opposite_label_is_not_a_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_prior([KTO_GOOD])
    _model_returns(monkeypatch, [KTO_BAD, KTO_GOOD, KTO_BAD])

    kept, _ = _generate(
        3, "--template", "preference", "--pref-task", "kto", "--dedup-with", "prior.jsonl"
    )

    assert kept == [KTO_BAD]


def test_distinct_preference_rows_are_all_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        DPO,
        {**DPO, "chosen": "four"},
        {**DPO, "rejected": "22"},
        {**DPO, "prompt": "What is 3+3?"},
    ]
    _write_prior([{"prompt": "unrelated", "chosen": "a", "rejected": "b"}])
    _model_returns(monkeypatch, rows)

    kept, output = _generate(4, "--template", "preference", "--dedup-with", "prior.jsonl")

    assert kept == rows
    assert "Duplicates" not in output


def test_alpaca_dedup_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {"instruction": "i", "output": "o"}
    _write_prior([row])
    _model_returns(monkeypatch, [row, {"instruction": "j", "output": "o"}])

    kept, output = _generate(2, "--template", "qa", "--dedup-with", "prior.jsonl")

    assert kept == [{"instruction": "j", "output": "o"}]
    assert "Duplicates: 1 removed" in output


def test_dedup_key_names_preference_fields_only() -> None:
    assert generate._dedup_key(DPO) == generate._dedup_key(
        {"source": "x", "rejected": "5", "prompt": "What is 2+2?", "chosen": "4"}
    )
    assert generate._dedup_key(KTO_GOOD) != generate._dedup_key(KTO_BAD)
    assert generate._dedup_key({"instruction": "i", "output": "o"}) == "i o"


def test_the_generate_docs_say_preference_rows_are_deduped() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "data.md").read_text(encoding="utf-8")
    section = text.split("## Synthetic Data Generation", 1)[1]
    section = " ".join(section.split("\n## ", 1)[0].split())
    assert "`--template preference` rows too" in section, section
    assert "`prompt` / `chosen` / `rejected`" in section, section
    assert "`prompt` / `completion` / `label`" in section, section
