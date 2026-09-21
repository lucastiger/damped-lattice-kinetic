"""YAML configuration loading and resolution."""

from __future__ import annotations

import pytest

from dlkin import Grid, LatticeModel, load_config, resolve

EXAMPLE = """
grid:
  L: 200.0
  N: 4096
model:
  gamma: 0.1
  mu: 1.0
  mu2: -0.5
  couplings: [[1, 1.0], [2, 0.25]]
  template_width: 2.0
run:
  c0: 0.88
  c_values: [0.881, 0.882]
"""


def test_load_and_resolve(tmp_path) -> None:
    path = tmp_path / "example.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")

    config = load_config(path)
    assert config["run"]["c0"] == 0.88

    resolved = resolve(config)
    assert resolved.grid == Grid(L=200.0, N=4096)
    assert resolved.model == LatticeModel(
        gamma=0.1, mu=1.0, mu2=-0.5, couplings=((1, 1.0), (2, 0.25)), template_width=2.0
    )
    assert resolved.run == {"c0": 0.88, "c_values": [0.881, 0.882]}


def test_model_section_falls_back_to_defaults(tmp_path) -> None:
    path = tmp_path / "minimal.yaml"
    path.write_text("grid: {L: 100.0, N: 256}\n", encoding="utf-8")
    resolved = resolve(load_config(path))
    assert resolved.model == LatticeModel()
    assert resolved.run == {}


def test_missing_grid_key_is_reported(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("grid: {L: 100.0}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing key 'N'"):
        resolve(load_config(path))
