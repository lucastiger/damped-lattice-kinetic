"""The claims checker: the arithmetic, and the three rules that matter more than it.

A claim that quietly does not get checked is worse than one that visibly fails, so the
cases exercised hardest here are the ones where something is *absent*: a missing file, an
unresolvable key, a data file that holds quick results.  None of those may pass, and none
may vanish from the accounting.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def cc():
    """Import scripts/check_claims.py as a module."""
    spec = importlib.util.spec_from_file_location("check_claims", SCRIPTS / "check_claims.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_claims"] = module
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, payload: dict, name: str = "data/x.json", quick: bool = False) -> None:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"metadata": {"quick": quick, "git_commit": "abc1234", "elapsed_sec": 1.0}, "quick": quick}
    document.update(payload)
    path.write_text(json.dumps(document), encoding="utf-8")


def _manifest(*claims: dict) -> dict:
    return {"schema": 1, "claims": [{"file": "data/x.json", **c} for c in claims]}


# ------------------------------------------------------------ the tests ----
@pytest.mark.parametrize(
    "claim, value, expect",
    [
        ({"kind": "value", "value": 1.0, "abs_tol": 1e-3}, 1.0005, True),
        ({"kind": "value", "value": 1.0, "abs_tol": 1e-3}, 1.002, False),
        ({"kind": "value", "value": 200.0, "rel_tol": 1e-2}, 201.0, True),
        ({"kind": "value", "value": 200.0, "rel_tol": 1e-2}, 210.0, False),
        ({"kind": "upper", "bound": 1e-9}, 5e-10, True),
        ({"kind": "upper", "bound": 1e-9}, 5e-9, False),
        ({"kind": "order", "value": 1e-8, "decades": 1.0}, 5e-8, True),
        ({"kind": "order", "value": 1e-8, "decades": 1.0}, 5e-6, False),
        ({"kind": "bool", "value": True}, True, True),
        ({"kind": "bool", "value": True}, False, False),
        ({"kind": "null"}, None, True),
        ({"kind": "null"}, 0.0, False),
    ],
)
def test_each_kind(cc, claim, value, expect) -> None:
    ok, _, _, _ = cc.apply_test(claim, value)
    assert ok is expect


def test_bool_is_identity_not_truthiness(cc) -> None:
    """`x is value`: 1 is not True, and a non-empty string is not True."""
    assert not cc.apply_test({"kind": "bool", "value": True}, 1)[0]
    assert not cc.apply_test({"kind": "bool", "value": True}, "yes")[0]


def test_a_null_value_fails_a_numeric_claim(cc) -> None:
    """A key that resolved to null cannot satisfy a value/upper/order claim."""
    for claim in (
        {"kind": "value", "value": 1.0, "abs_tol": 1.0},
        {"kind": "upper", "bound": 1.0},
        {"kind": "order", "value": 1.0, "decades": 9.0},
    ):
        ok, _, _, detail = cc.apply_test(claim, None)
        assert not ok
        assert "null" in detail


def test_nan_fails(cc) -> None:
    assert not cc.apply_test({"kind": "upper", "bound": 1.0}, float("nan"))[0]


def test_order_rejects_exact_zero(cc) -> None:
    ok, _, _, detail = cc.apply_test({"kind": "order", "value": 1e-8, "decades": 1.0}, 0.0)
    assert not ok and "log10" in detail


def test_missing_file_is_a_failure_never_a_skip(cc, tmp_path) -> None:
    manifest = _manifest({"id": "A", "key": "a", "kind": "upper", "bound": 1.0})
    results, files = cc.check(manifest, tmp_path)
    assert [r.status for r in results] == [cc.FAIL]
    assert "missing" in results[0].detail
    assert files["data/x.json"]["exists"] is False


def test_unresolvable_key_is_a_failure(cc, tmp_path) -> None:
    _write(tmp_path, {"a": {"b": 1.0}})
    manifest = _manifest({"id": "A", "key": "a/nope", "kind": "upper", "bound": 1.0})
    results, _ = cc.check(manifest, tmp_path)
    assert results[0].status == cc.FAIL
    assert "did not resolve" in results[0].detail


def test_quick_data_is_skipped_loudly_and_strict_fails_it(cc, tmp_path) -> None:
    _write(tmp_path, {"a": 0.5}, quick=True)
    manifest = _manifest({"id": "A", "key": "a", "kind": "upper", "bound": 1.0})
    results, _ = cc.check(manifest, tmp_path)
    assert results[0].status == cc.SKIPPED_QUICK
    # the value WOULD have passed; it is skipped because the data is not claim-grade
    assert cc._summary(results, strict=False) == {
        "total": 1, "passed": 0, "failed": 0, "skipped_quick": 1, "exact_replication_failures": 0,
    }
    assert cc._summary(results, strict=True)["failed"] == 1


def test_accounting_is_exhaustive(cc, tmp_path) -> None:
    """passed + failed + skipped must equal the manifest size, in both modes."""
    _write(tmp_path, {"good": 0.5, "bad": 5.0})
    _write(tmp_path, {"good": 0.5}, name="data/q.json", quick=True)
    manifest = {
        "claims": [
            {"file": "data/x.json", "id": "OK", "key": "good", "kind": "upper", "bound": 1.0},
            {"file": "data/x.json", "id": "NO", "key": "bad", "kind": "upper", "bound": 1.0},
            {"file": "data/x.json", "id": "GONE", "key": "absent", "kind": "upper", "bound": 1.0},
            {"file": "data/q.json", "id": "QUICK", "key": "good", "kind": "upper", "bound": 1.0},
            {"file": "data/missing.json", "id": "NOFILE", "key": "a", "kind": "upper", "bound": 1.0},
        ]
    }
    results, _ = cc.check(manifest, tmp_path)
    for strict in (False, True):
        c = cc._summary(results, strict)
        assert c["passed"] + c["failed"] + c["skipped_quick"] == c["total"] == 5


def test_exact_replication_failures_are_flagged(cc, tmp_path) -> None:
    _write(tmp_path, {"a": 5.0})
    manifest = _manifest(
        {"id": "DRIFT", "key": "a", "kind": "upper", "bound": 1.0, "exact_replication": True}
    )
    results, _ = cc.check(manifest, tmp_path)
    assert results[0].status == cc.FAIL and results[0].exact_replication
    assert cc._summary(results, strict=False)["exact_replication_failures"] == 1


def test_exit_code_and_report(cc, tmp_path, capsys) -> None:
    """Exit 0 only when nothing failed; the report and the JSON both get written."""
    _write(tmp_path, {"a": 0.5})
    manifest_path = tmp_path / "claims.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_manifest({"id": "A", "key": "a", "kind": "upper", "bound": 1.0, "where": "Table 1"})),
        encoding="utf-8",
    )
    report, out_json = tmp_path / "r.md", tmp_path / "r.json"
    code = cc.main(
        ["--manifest", str(manifest_path), "--data-root", str(tmp_path),
         "--report", str(report), "--json", str(out_json)]
    )
    assert code == 0
    text = report.read_text(encoding="utf-8")
    assert "## UNRESOLVED" in text and "None." in text
    assert "Table 1" in text
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["passed"] == 1
    assert payload["claims"][0]["id"] == "A"


def test_exit_code_is_nonzero_on_failure(cc, tmp_path) -> None:
    _write(tmp_path, {"a": 5.0})
    manifest_path = tmp_path / "claims.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_manifest({"id": "A", "key": "a", "kind": "upper", "bound": 1.0})),
        encoding="utf-8",
    )
    report = tmp_path / "r.md"
    code = cc.main(
        ["--manifest", str(manifest_path), "--data-root", str(tmp_path), "--report", str(report)]
    )
    assert code == 1
    # UNRESOLVED comes first in the report, before anything reassuring
    text = report.read_text(encoding="utf-8")
    assert text.index("## UNRESOLVED") < text.index("## Summary")
    assert "did not pass" in text


def test_strict_makes_quick_data_nonzero_exit(cc, tmp_path) -> None:
    _write(tmp_path, {"a": 0.5}, quick=True)
    manifest_path = tmp_path / "claims.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_manifest({"id": "A", "key": "a", "kind": "upper", "bound": 1.0})),
        encoding="utf-8",
    )
    assert cc.main(["--manifest", str(manifest_path), "--data-root", str(tmp_path)]) == 0
    assert cc.main(["--manifest", str(manifest_path), "--data-root", str(tmp_path), "--strict"]) == 1


def test_manuscript_copy_is_preferred_and_matches_the_handoff_original(cc) -> None:
    """The manifest travels with the paper; the two copies must not drift apart."""
    path, manifest = cc.load_manifest(None)
    assert path.name == "claims.yaml" and path.parent.name == "manuscript"
    assert cc.manifests_differ() is None
    assert len(manifest["claims"]) > 100
