from __future__ import annotations

import pytest

from safety_platform.evaluation.__main__ import build_parser

_JBB_V2_FETCH_OUT = "evaluation/data/jbb-v2"


def test_run_speculative_echo_flag_defaults_off_and_wires_true_on_flag() -> None:
    parser = build_parser()

    off = parser.parse_args(["run"])
    assert off.speculative_echo is False

    on = parser.parse_args(["run", "--speculative-echo"])
    assert on.speculative_echo is True


def test_build_parser_pins_jbb_v2_defaults(tmp_path, capsys):  # type: ignore[no-untyped-def]
    parser = build_parser()

    fetch = parser.parse_args(["fetch", "--out", str(tmp_path)])
    assert fetch.out == tmp_path
    run = parser.parse_args(
        ["run", "--dataset", str(tmp_path / "cases.jsonl"), "--db", str(tmp_path)]
    )
    assert run.reuse_run is None
    assert run.per_label == 20

    defaults = parser.parse_args(["run"])
    assert str(defaults.dataset).endswith("jbb-v2/cases.jsonl")
    assert str(defaults.db).endswith("jbb-v2.sqlite3")
    assert defaults.target == "llama3.2:latest"
    assert defaults.tier3 == "qwen3:8b"
    assert defaults.judge == "qwen3:8b"

    report = parser.parse_args(["report"])
    assert str(report.db).endswith("jbb-v2.sqlite3")


def test_fetch_defaults_to_frozen_jbb_v2_directory() -> None:
    parser = build_parser()
    arguments = parser.parse_args(["fetch"])
    assert str(arguments.out) == _JBB_V2_FETCH_OUT


def test_fetch_force_flag_defaults_off_and_wires_overwrite(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    parser = build_parser()
    assert parser.parse_args(["fetch"]).force is False

    captured: dict[str, object] = {}

    def fake_fetch(out, *, train_per_label, test_per_label, overwrite):  # type: ignore[no-untyped-def]
        captured["overwrite"] = overwrite
        return out

    monkeypatch.setattr(
        "safety_platform.evaluation.__main__.fetch_jbb_dataset", fake_fetch
    )
    from safety_platform.evaluation import __main__ as cli

    cli.main(["fetch", "--force"])
    assert captured["overwrite"] is True

    cli.main(["fetch"])
    assert captured["overwrite"] is False


@pytest.mark.parametrize("argv", [["fetch", "--train-per-label", "0"]])
def test_fetch_rejects_non_positive_split_size(argv: list[str], tmp_path) -> None:  # type: ignore[no-untyped-def]
    from safety_platform.evaluation import __main__ as cli

    with pytest.raises(ValueError, match="split sizes must be positive"):
        cli.main([*argv, "--out", str(tmp_path)])
