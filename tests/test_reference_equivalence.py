"""THE GATE: the spectral / biorthogonal layer must equal the reference, and the note.

Two kinds of check live here.

*Live equivalence* (fast): run ``handoff/reference/spectral.py`` and ``dlkin.spectral``
side by side on the same branch and require them to agree.  This catches any drift in the
method itself -- a different ``phat`` normalization, a missing Nyquist zeroing, a different
continuation path, finite differences sneaking into ``sigma'``.

*Ground truth* (slow): reproduce the numbers in ``handoff/reference/verify_core.json`` and
``verify_extra.json``, which were recomputed from scratch after a sandbox reset and match
the manuscript to all quoted digits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from dlkin import (
    Grid,
    LatticeModel,
    TravelingWaveSolver,
    biorthogonal_scalars,
    classify_eigs,
    drop_arrays,
    init_branch,
    natural_continuation,
    pencil_eigs,
)

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "handoff" / "reference"

#: The march verify_core.py performs after init_branch, before reading scalars at c = 0.89.
MARCH = tuple(np.round(np.arange(0.881, 0.8901, 0.001), 6))
#: The rows verify_core.py's block (c) then walks, in order.  The path matters: it is what
#: pins which branch the solutions sit on.
THRESHOLD_ROWS = (0.8988, 0.89892, 0.8990)


def _reference(name: str):
    """Import a module from handoff/reference/ without leaving it on sys.path."""
    sys.path.insert(0, str(REFERENCE_DIR))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(REFERENCE_DIR))


@pytest.fixture(scope="module")
def verify_core() -> dict:
    return json.loads((REFERENCE_DIR / "verify_core.json").read_text())


@pytest.fixture(scope="module")
def verify_extra() -> dict:
    return json.loads((REFERENCE_DIR / "verify_extra.json").read_text())


def _relative(value: float, target: float) -> float:
    return abs(value - target) / abs(target)


# ----------------------------------------------------------------- live equivalence ---
@pytest.mark.parametrize("N", [1024, 2048])
def test_scalars_match_reference_spectral(N: int) -> None:
    """dlkin.spectral vs handoff/reference/spectral.py at c = 0.89, same branch path."""
    fk = _reference("fk")
    reference_spectral = _reference("spectral")
    L, c = 200.0, 0.89

    ref_solver, ref_psi, ref_sigma = fk.init_branch(L, N)
    for velocity in MARCH:
        ref_psi, ref_sigma, _, _ = ref_solver.solve(velocity, ref_psi, ref_sigma)
    expected = reference_spectral.scalars(ref_solver, ref_psi, ref_sigma, c)

    solver, psi, drive = init_branch(Grid(L=L, N=N), LatticeModel())
    records = natural_continuation(solver, psi, drive, MARCH)
    got = biorthogonal_scalars(
        solver, records[-1].psi, records[-1].drive, c, phat_mode="minus2pi"
    )

    assert got["phat_mode"] == "minus2pi"
    for mine, theirs in (
        ("sigma", "sigma"),
        ("sigma_prime", "sigp"),
        ("kappa", "kappa"),
        ("m", "m"),
        ("nu2_pred", "nu2_pred"),
    ):
        assert _relative(got[mine], expected[theirs]) < 1e-10, (N, mine)

    # the supporting quantities too, on the same footing
    assert _relative(got["phat_one"], expected["one"]) < 1e-10
    assert _relative(got["dcphi_edge"], expected["dcpsi_bdry"]) < 1e-10
    assert _relative(got["power_balance"], expected["sigma_pb"] * solver.model.mu) < 1e-10


def test_phat_normalizations_differ_but_ratios_do_not() -> None:
    """The two conventions give different kappa and <phat,1>, the same -kappa/m.

    This is the confusion the API is designed to prevent, so it is pinned by a test.
    """
    solver, psi, drive = init_branch(Grid(L=200.0, N=1024), LatticeModel())
    records = natural_continuation(solver, psi, drive, MARCH)
    psi, drive = records[-1].psi, records[-1].drive

    minus2pi = biorthogonal_scalars(solver, psi, drive, 0.89, phat_mode="minus2pi")
    unit = biorthogonal_scalars(solver, psi, drive, 0.89, phat_mode="unit")

    assert minus2pi["phat_one"] == pytest.approx(-2.0 * np.pi, abs=1e-12)
    assert abs(unit["phat_one"]) < abs(minus2pi["phat_one"])
    assert _relative(unit["kappa"], minus2pi["kappa"]) > 0.5  # genuinely different

    # ratios, and the identity, are convention-independent
    assert _relative(unit["nu2_pred"], minus2pi["nu2_pred"]) < 1e-10
    assert _relative(unit["identity_relerr"], minus2pi["identity_relerr"]) < 1e-6
    for record in (minus2pi, unit):
        scale = record["kappa"] / record["phat_one"]
        assert scale == pytest.approx(-record["drive_prime"], rel=1e-3)


def test_phat_mode_is_required() -> None:
    """No default: a caller must state which convention they mean."""
    from dlkin.spectral import normalize_phat

    with pytest.raises(TypeError):
        normalize_phat(np.ones(4), np.ones(4), Grid(L=1.0, N=4))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="phat_mode must be one of"):
        solver, psi, drive = init_branch(Grid(L=50.0, N=256), LatticeModel(), c0=0.6)
        biorthogonal_scalars(solver, psi, drive, 0.6, phat_mode="l2")


def test_exact_derivative_beats_finite_differences() -> None:
    """sigma'(c) by bordered solve, not by differencing -- and the gap is visible.

    SCIENCE_BRIEF sec. 4 warns that a centred difference with step 1e-3 carries a
    grid-independent error of order 1e-4 where sigma(c) curves hardest.  Measured here at
    c = 0.89, N = 1024, the difference from the exact bordered solve is 1.2e-3 -- eleven
    orders of magnitude above the 1e-14 at which the exact value matches the reference.
    The point of the test is that margin, so nobody is tempted to swap the solve out.
    """
    from dlkin.spectral import exact_drive_derivative

    solver, psi, drive = init_branch(Grid(L=200.0, N=1024), LatticeModel())
    records = natural_continuation(solver, psi, drive, MARCH)
    psi, drive = records[-1].psi, records[-1].drive
    c, step = 0.89, 1e-3

    _, exact = exact_drive_derivative(solver, psi, drive, c)

    forward = natural_continuation(solver, psi, drive, [c + step])[-1]
    backward = natural_continuation(solver, psi, drive, [c - step])[-1]
    differenced = (forward.drive - backward.drive) / (2.0 * step)

    assert 1e-5 < abs(exact - differenced) < 1e-2


# --------------------------------------------------------------------- ground truth ---
@pytest.fixture(scope="module")
def core_4096():
    """verify_core.py's own path at L = 200, N = 4096: init_branch, march, block (c)."""
    solver, psi, drive = init_branch(Grid(L=200.0, N=4096), LatticeModel())
    records = natural_continuation(solver, psi, drive, MARCH)
    psi, drive = records[-1].psi, records[-1].drive

    at_089 = drop_arrays(biorthogonal_scalars(solver, psi, drive, 0.89, "minus2pi"))

    rows = {}
    for c in THRESHOLD_ROWS:
        record = natural_continuation(solver, psi, drive, [c])[-1]
        psi, drive = record.psi, record.drive
        full = biorthogonal_scalars(solver, psi, drive, c, "minus2pi")
        # keep the matrices only for the row the Arnoldi test needs
        rows[c] = full if c == THRESHOLD_ROWS[-1] else drop_arrays(full)
    return {"at_089": at_089, "rows": rows}


@pytest.mark.slow
def test_verify_core_identity_block(core_4096, verify_core) -> None:
    """verify_core.json block (b): the Jordan-chain identity at c = 0.89, N = 4096."""
    got = core_4096["at_089"]
    target = verify_core["b"]

    assert _relative(got["sigma"], 0.6373006910107105) < 1e-12
    assert _relative(got["sigma_prime"], 1.9251320609658795) < 1e-10
    assert _relative(got["kappa"], 12.095961479841174) < 1e-10
    assert _relative(got["m"], 207.51277258293618) < 1e-8
    assert _relative(got["nu2_pred"], -0.05829020223324715) < 1e-8

    assert got["identity_relerr"] < 1e-13
    assert got["power_balance_relerr"] < 1e-12
    assert abs(got["phat_dot_Vpp"]) < 1e-11

    # the remaining entries of block (b), on the reference's own tolerances
    assert got["phat_one"] == pytest.approx(-2.0 * np.pi, abs=1e-12)
    assert _relative(got["dcphi_edge"], target["dcpsi_bdry"]) < 1e-10
    assert _relative(got["dcphi_edge_pred"], target["dinf_pred"]) < 1e-10
    assert got["res_phat_rel"] < 1e-12
    assert _relative(got["res_M0_phiprime"], target["res_M0p0"]) < 1e-2


@pytest.mark.slow
def test_verify_core_threshold_row_0899(core_4096) -> None:
    """verify_core.json block (c) at c = 0.8990, reached by the reference's own path.

    Tolerance note: run live in this environment, ``handoff/reference/spectral.py`` on this
    same path returns these values BITWISE, so the ~6e-11 residual against the stored JSON
    is drift in the stored numbers (a different BLAS summation order), not a difference of
    method.  The branch is close to the fold here and ``M0`` is nearly singular, which is
    what amplifies round-off into the eleventh digit.
    """
    got = core_4096["rows"][0.8990]
    assert _relative(got["sigma_prime"], -0.09005272389171688) < 1e-9
    assert _relative(got["kappa"], -0.5658179516125822) < 1e-9
    assert _relative(got["m"], 290.4159271788894) < 1e-8
    assert _relative(got["nu2_pred"], 0.0019483020683781287) < 1e-9

    # sigma' changes sign between 0.89892 and 0.8990: c_hat1 is bracketed
    assert core_4096["rows"][0.89892]["sigma_prime"] > 0.0
    assert got["sigma_prime"] < 0.0


@pytest.mark.slow
def test_verify_core_arnoldi_eigenvalue(core_4096) -> None:
    """verify_core.json block (d): the unstable real eigenvalue at c = 0.8990."""
    row = core_4096["rows"][0.8990]
    shift = max(row["nu2_pred"], 0.0) + 0.004
    nus = pencil_eigs(row["M0"], row["M1"], shift=shift, k=24)
    classified = classify_eigs(nus, LatticeModel().gamma)

    candidates = [
        value
        for value in classified["real_nontrivial"]
        if abs(value - row["nu2_pred"]) < 0.05
    ]
    assert candidates, classified["real_nontrivial"]
    assert candidates[0] == pytest.approx(0.001931469, abs=1e-7)
    assert candidates[0] > 0.0  # unstable, rho = exp(nu/c) > 1
    assert np.exp(candidates[0] / 0.8990) == pytest.approx(1.00215, abs=2e-5)
    assert classified["has_zero"] is True


@pytest.mark.slow
def test_spectrum_snapshot_pairing_and_reality() -> None:
    """The duality R2 and the absence of real modes at c = 0.89, as claims.yaml has them."""
    solver, psi, drive = init_branch(Grid(L=200.0, N=2048), LatticeModel())
    records = natural_continuation(solver, psi, drive, MARCH)
    scalars = biorthogonal_scalars(
        solver, records[-1].psi, records[-1].drive, 0.89, "minus2pi"
    )
    nus = pencil_eigs(scalars["M0"], scalars["M1"], shift=-0.045, k=40)
    classified = classify_eigs(nus, 0.1)

    assert classified["has_zero"] is True
    assert classified["has_minus_gamma"] is True
    assert classified["real_nontrivial"] == []
    assert classified["pairing_max_err"] < 1e-7

    # R3 says the constant-coefficient spectrum is exactly Re nu = -gamma/2, i.e. the
    # multiplier circle |rho| = exp(-gamma/(2c)).  On a finite window with the kink's
    # localized potential the computed approximants do not sit on that line individually:
    # they scatter by up to 3.8e-3 in Re nu, in pairs symmetric about -gamma/2 (which is
    # the duality R2 again, and is what pairing_max_err above measures).  What is sharp is
    # the circle they straddle -- claims.yaml SP-circle, 0.945369 to 1e-6.
    essential = [x for x in nus if abs(x) > 1e-6 and abs(x + 0.1) > 1e-6]
    assert len(essential) > 20
    deviation = [abs(x.real + 0.05) for x in essential]
    assert max(deviation) < 1e-2
    circle = float(np.median(np.abs(np.exp(np.asarray(essential) / 0.89))))
    assert circle == pytest.approx(float(np.exp(-0.05 / 0.89)), abs=1e-6)
    assert circle == pytest.approx(0.945369, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("N", [2048, 4096])
def test_verify_extra_generalized_model(N: int, verify_extra: dict) -> None:
    """verify_extra.json: kappa = -f'(c) <phat,1> for the second-harmonic lattice (R8).

    ``gen.py`` starts from a flat ``psi`` at ``c = 0.55`` -- no two-stage init_branch --
    and uses the Euclidean (``"unit"``) normalization of ``phat``.  That normalization is
    grid-dependent, which is why the reference ``kappa`` changes by ``sqrt(2)`` between
    ``N = 2048`` and ``N = 4096`` while ``f`` and ``f'`` barely move; the identity's
    relative error does not care.
    """
    target = verify_extra[f"mu2m05_N{N}"]
    grid = Grid(L=200.0, N=N)
    model = LatticeModel(gamma=0.1, mu=1.0, mu2=-0.5, couplings=((1, 1.0),))
    solver = TravelingWaveSolver(grid, model)

    psi, drive = np.zeros(N), 0.5
    for c in (0.55, 0.65, 0.75, 0.82):
        result = solver.newton(c, psi, drive, tol=1e-11)  # gen.py's tolerance
        psi, drive = result.psi, result.drive

    got = biorthogonal_scalars(solver, psi, drive, 0.82, phat_mode="unit")
    assert got["phat_mode"] == "unit"
    assert _relative(got["drive"], target["f"]) < 1e-10
    assert _relative(got["drive_prime"], target["fp"]) < 1e-10
    assert _relative(got["kappa"], target["kappa"]) < 1e-10

    if N == 2048:
        # order of magnitude: the identity's discretization error at h ~ 0.195
        assert 1e-6 < got["identity_relerr"] < 1e-5
        assert abs(np.log10(got["identity_relerr"] / 3.93e-6)) < 0.5
    else:
        assert got["identity_relerr"] < 1e-10
