"""Result I/O: provenance, CSV tables, and claims-style key lookup."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dlkin import (
    Grid,
    LatticeModel,
    get,
    load_result,
    resolve,
    save_result,
    save_table_csv,
    timer,
)
from dlkin.config import ResolvedConfig, RunConfig
from dlkin.io import collect_meta, git_commit, to_jsonable


@pytest.fixture
def resolved() -> ResolvedConfig:
    return resolve(
        {
            "grid": {"L": 200.0, "N": 1024},
            "model": {"gamma": 0.1, "mu": 1.0, "couplings": [[1, 1.0]]},
            "run": {
                "name": "identity",
                "c0": 0.88,
                "c_values": {"start": 0.881, "stop": 0.8831, "step": 0.001},
                "phat_mode": "minus2pi",
                "tolerances": {"newton": 1e-12},
                "arnoldi": {"k": 24},
                "shifts": [0.004],
            },
        }
    )


def test_save_result_records_provenance(tmp_path, resolved) -> None:
    with timer() as elapsed:
        payload = {"c089": {"kappa": np.float64(12.0959614798), "phat_mode": "minus2pi"}}
    path = save_result(
        tmp_path / "01_kinetic.json", payload, {"config": resolved.as_meta()},
        wall_time=elapsed(),
    )
    document = load_result(path)

    meta = document["meta"]
    for key in (
        "timestamp",
        "git_commit",
        "python",
        "numpy",
        "scipy",
        "hostname",
        "platform",
        "wall_time_seconds",
        "config",
    ):
        assert key in meta, key
    assert meta["git_commit"] == git_commit()
    assert meta["config"]["grid"] == {"L": 200.0, "N": 1024, "h": 200.0 * 2 / 1024}
    assert meta["config"]["model"]["gamma"] == 0.1
    assert meta["config"]["run"]["phat_mode"] == "minus2pi"
    assert meta["config"]["run"]["c_values"] == [0.881, 0.882, 0.883]
    assert meta["config"]["run"]["params"]["shifts"] == [0.004]
    assert document["c089"]["kappa"] == pytest.approx(12.0959614798)


def test_save_result_requires_a_config(tmp_path) -> None:
    """A result without its configuration is not reproducible, so it is not writable."""
    with pytest.raises(ValueError, match="must include a 'config' entry"):
        save_result(tmp_path / "x.json", {"a": 1}, {"note": "forgot"})


def test_save_result_rejects_a_meta_key_in_the_payload(tmp_path, resolved) -> None:
    with pytest.raises(ValueError, match="must not contain a 'meta' key"):
        save_result(tmp_path / "x.json", {"meta": 1}, {"config": resolved.as_meta()})


def test_meta_caller_values_win(resolved) -> None:
    meta = collect_meta({"config": resolved.as_meta(), "hostname": "pinned"}, wall_time=1.5)
    assert meta["hostname"] == "pinned"
    assert meta["wall_time_seconds"] == 1.5


def test_to_jsonable_handles_numpy_and_complex() -> None:
    converted = to_jsonable(
        {
            "array": np.arange(3, dtype=float),
            "scalar": np.float64(2.5),
            "integer": np.int64(7),
            "complex": 1 + 2j,
            "nested": [np.bool_(True), (np.float32(1.5),)],
        }
    )
    assert json.dumps(converted)  # round-trips through the encoder
    assert converted["array"] == [0.0, 1.0, 2.0]
    assert converted["complex"] == [1.0, 2.0]


def test_save_table_csv_from_mappings(tmp_path) -> None:
    rows = [
        {"c": 0.895, "kappa": 9.6329939, "m": 209.0},
        {"c": 0.898, "kappa": 4.6609061},  # 'm' missing on purpose
    ]
    path = save_table_csv(tmp_path / "t.csv", rows, ["c", "kappa", "m"])
    lines = path.read_text().splitlines()
    assert lines[0] == "c,kappa,m"
    assert lines[1] == "0.895,9.6329939,209.0"
    assert lines[2] == "0.898,4.6609061,"


def test_save_table_csv_from_sequences(tmp_path) -> None:
    path = save_table_csv(tmp_path / "t.csv", [[1, 2], [3, 4]], ["a", "b"])
    assert path.read_text().splitlines() == ["a,b", "1,2", "3,4"]
    with pytest.raises(ValueError, match="row 0 has 3 entries"):
        save_table_csv(tmp_path / "u.csv", [[1, 2, 3]], ["a", "b"])


def test_get_uses_slash_paths_and_tolerates_dots_in_keys() -> None:
    """claims.yaml addresses rows by velocity, e.g. 'rows/0.89500/kappa'."""
    payload = {"rows": {"0.89500": {"kappa": 9.6329939}}, "list": [{"x": 1}]}
    assert get(payload, "rows/0.89500/kappa") == 9.6329939
    assert get(payload, "list/0/x") == 1
    assert get(payload, "rows") == {"0.89500": {"kappa": 9.6329939}}


def test_get_reports_a_missing_key_clearly() -> None:
    payload = {"rows": {"0.89500": {"kappa": 1.0}}}
    with pytest.raises(KeyError, match="no key 'sigma'"):
        get(payload, "rows/0.89500/sigma")
    with pytest.raises(KeyError, match="no key '0.90000'"):
        get(payload, "rows/0.90000/kappa")
    with pytest.raises(KeyError, match="cannot descend into float"):
        get(payload, "rows/0.89500/kappa/deeper")


def test_claims_yaml_paths_are_all_parseable() -> None:
    """Every key in claims.yaml is a '/'-separated path this lookup can walk."""
    import yaml

    from pathlib import Path

    manifest = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "handoff" / "claims.yaml").read_text()
    )
    for claim in manifest["claims"]:
        parts = claim["key"].split("/")
        assert all(parts), claim["id"]
        nested: dict = {}
        cursor = nested
        for part in parts[:-1]:
            cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = "value"
        assert get(nested, claim["key"]) == "value", claim["id"]


def test_run_config_resolution() -> None:
    resolved = resolve({"grid": {"L": 100.0, "N": 256}, "run": {"c_values": [0.5, 0.6]}})
    assert resolved.run_config.c_values == (0.5, 0.6)
    assert resolved.run_config.phat_mode == "minus2pi"
    assert resolved.grid == Grid(L=100.0, N=256)
    assert resolved.model == LatticeModel()

    with pytest.raises(ValueError, match="phat_mode must be one of"):
        RunConfig(phat_mode="l2")
