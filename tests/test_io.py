"""Result files: the metadata block, key-path access, merging and the config overlay."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dlkin import (
    apply_overlay,
    deep_merge,
    get_path,
    load_config,
    load_result,
    save_result,
    set_path,
    to_jsonable,
)


def test_payload_sits_at_the_top_level_next_to_metadata(tmp_path) -> None:
    """claims.yaml addresses `critical/c_hat1`, so nothing may wrap the payload."""
    out = tmp_path / "r.json"
    save_result(out, {"critical": {"c_hat1": 0.8989297}}, script="s.py")
    document = load_result(out)
    assert get_path(document, "critical/c_hat1") == 0.8989297
    assert document["metadata"]["script"] == "s.py"
    assert document["metadata"]["schema"] == 1


def test_quick_is_recorded_in_both_places(tmp_path) -> None:
    """A quick run is at reduced L/N and must be skipped by the checker."""
    out = tmp_path / "r.json"
    save_result(out, {"x": 1}, script="s.py", quick=True)
    document = load_result(out)
    assert document["quick"] is True
    assert document["metadata"]["quick"] is True


def test_numpy_scalars_and_arrays_are_written(tmp_path) -> None:
    out = tmp_path / "r.json"
    save_result(out, {"a": np.float64(1.5), "b": np.arange(3), "c": {"d": np.int64(7)}}, script="s.py")
    document = load_result(out)
    assert document["a"] == 1.5
    assert document["b"] == [0, 1, 2]
    assert document["c"]["d"] == 7


def test_merge_keeps_sections_from_earlier_runs(tmp_path) -> None:
    """An expensive `--only` section folds into an existing file rather than replacing it."""
    out = tmp_path / "r.json"
    save_result(out, {"convergence": {"L200_N1024": {"relerr": 3e-4}}}, script="s.py", sections=["convergence"])
    save_result(
        out,
        {"convergence": {"L300_N6144": {"relerr": 1e-12}}},
        script="s.py",
        sections=["convergence"],
        merge=True,
    )
    document = load_result(out)
    assert get_path(document, "convergence/L200_N1024/relerr") == 3e-4
    assert get_path(document, "convergence/L300_N6144/relerr") == 1e-12
    assert len(document["metadata"]["previous_runs"]) == 1


def test_merge_makes_quick_sticky(tmp_path) -> None:
    """One quick section poisons the file: it no longer reproduces the claims as a whole."""
    out = tmp_path / "r.json"
    save_result(out, {"a": 1}, script="s.py", quick=True)
    save_result(out, {"b": 2}, script="s.py", quick=False, merge=True)
    assert load_result(out)["quick"] is True


def test_reserved_keys_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        save_result(tmp_path / "r.json", {"metadata": {}}, script="s.py")


def test_get_path_names_where_it_diverged() -> None:
    with pytest.raises(KeyError, match="rows"):
        get_path({"rows": {"0.89500": {}}}, "rows/0.89500/kappa")


def test_set_path_creates_intermediate_levels() -> None:
    data: dict = {}
    set_path(data, "rows/0.89500/kappa", 9.63)
    assert data == {"rows": {"0.89500": {"kappa": 9.63}}}


def test_deep_merge_merges_mappings_and_replaces_the_rest() -> None:
    base = {"grid": {"L": 200.0, "N": 4096}, "run": {"c": 0.89}}
    assert deep_merge(base, {"grid": {"N": 1024}}) == {
        "grid": {"L": 200.0, "N": 1024},
        "run": {"c": 0.89},
    }


def test_quick_overlay_replaces_only_what_it_names(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "grid: {L: 200.0, N: 4096}\nrun: {c: 0.89, step: 0.001}\n"
        "quick:\n  grid: {L: 100.0, N: 1024}\n  run: {step: 0.01}\n",
        encoding="utf-8",
    )
    config = load_config(path)
    quick = apply_overlay(config, "quick")
    assert "quick" not in quick
    assert quick["grid"] == {"L": 100.0, "N": 1024}
    assert quick["run"] == {"c": 0.89, "step": 0.01}


def test_to_jsonable_leaves_plain_values_alone() -> None:
    assert to_jsonable({"a": [1, None, True, "x"]}) == {"a": [1, None, True, "x"]}


def test_written_file_is_valid_json_with_a_trailing_newline(tmp_path) -> None:
    out = tmp_path / "r.json"
    save_result(out, {"x": 1}, script="s.py")
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["x"] == 1
