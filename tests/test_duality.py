"""The two operator identities the whole note rests on, tested directly.

R2: ``Q(-gamma - nu) = Q(nu)^T`` exactly, where ``Q(nu) = nu^2 I + nu M1 + M0``.
R1: ``A^T J + J A = -gamma J`` for ``A = [[0, I], [K, -gamma I]]`` with ``K = K^T``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlkin import (
    Grid,
    LatticeModel,
    circulant_from_symbol,
    pencil_matrix,
    symplectic_form,
    sym_d1,
    sym_shift_laplacian,
)

NUS = (0.0, 0.31, -0.1, -0.05 + 0.4j, 0.7 - 1.3j, 2.5 + 2.5j)


@pytest.fixture(scope="module")
def operators():
    """M0, M1 on a small grid at an arbitrary smooth (non-solution) profile."""
    grid = Grid(L=20.0, N=256)
    model = LatticeModel(gamma=0.1, mu=1.0, mu2=-0.3, couplings=((1, 1.0), (2, 0.25)))
    c = 0.77
    k, N = grid.k, grid.N
    Lin = circulant_from_symbol(model.lin_symbol(k, N, c), N)
    M1 = circulant_from_symbol(model.M1_symbol(k, N, c), N)
    # an arbitrary smooth phi -- the identity is structural and holds for any profile
    phi = 1.7 + 0.9 * np.sin(2.0 * np.pi * grid.xi / (2.0 * grid.L)) + 0.4 * np.cos(
        6.0 * np.pi * grid.xi / (2.0 * grid.L)
    )
    M0 = Lin + np.diag(model.Vpp(phi))
    return M0, M1, model.gamma


def test_pencil_duality(operators) -> None:
    """Q(-gamma-nu) - Q(nu)^T vanishes for real and complex nu alike."""
    M0, M1, gamma = operators
    scale = np.max(np.abs(M0))
    for nu in NUS:
        lhs = pencil_matrix(M0, M1, -gamma - nu)
        rhs = pencil_matrix(M0, M1, nu).T
        assert np.max(np.abs(lhs - rhs)) < 1e-12, (nu, scale)


def test_pencil_duality_needs_the_operator_symmetries(operators) -> None:
    """It holds because d_xi^T = -d_xi and Delta_j^T = Delta_j; check those directly."""
    M0, M1, gamma = operators
    grid = Grid(L=20.0, N=256)
    D1 = circulant_from_symbol(sym_d1(grid.k, grid.N), grid.N)
    assert np.max(np.abs(D1.T + D1)) < 1e-12
    for j in (1, 2):
        Dj = circulant_from_symbol(sym_shift_laplacian(grid.k, j), grid.N)
        assert np.max(np.abs(Dj.T - Dj)) < 1e-12
    # M0^T is M0 with gamma -> -gamma, i.e. the anti-damped operator
    assert np.max(np.abs(M0.T - (M0 + 2.0 * gamma * 0.77 * D1))) < 1e-12
    # M1^T = -M1 + 2 gamma I
    assert np.max(np.abs(M1.T - (-M1 + 2.0 * gamma * np.eye(grid.N)))) < 1e-12


def test_nyquist_zeroing_is_forced_not_chosen() -> None:
    """There is no real circulant with symbol ``i k``; the zeroing is the only option.

    The duality needs ``d_xi^T = -d_xi``, which needs a real circulant.  Keeping the
    Nyquist entry leaves ``ifft(i k)`` with an imaginary part of size ``|k_Nyq| / N`` --
    far above any round-off threshold -- so :func:`circulant_from_symbol` refuses it.  And
    quietly taking the real part is not a way around the convention: the real part is
    *exactly* the Nyquist-zeroed matrix, so a careless build silently imposes the same
    choice without recording it.
    """
    grid = Grid(L=20.0, N=256)
    unzeroed = 1j * np.asarray(grid.k)
    column = np.fft.ifft(unzeroed)

    assert np.max(np.abs(column.imag)) == pytest.approx(
        abs(grid.k[grid.N // 2]) / grid.N, rel=1e-12
    )
    with pytest.raises(ValueError, match="conjugate-symmetric"):
        circulant_from_symbol(unzeroed, grid.N)

    from scipy.linalg import circulant

    zeroed = circulant_from_symbol(sym_d1(grid.k, grid.N), grid.N)
    assert np.max(np.abs(circulant(column.real) - zeroed)) < 1e-15


@pytest.mark.parametrize("n", [4, 8, 16])
def test_conformal_symplectic_generator_identity(n: int) -> None:
    """A^T J + J A = -gamma J for A = [[0, I], [K, -gamma I]] with symmetric K."""
    rng = np.random.default_rng(7)
    gamma = 0.37
    J = symplectic_form(n)
    for _ in range(5):
        K = rng.standard_normal((n, n))
        K = (K + K.T) / 2.0
        A = np.block([[np.zeros((n, n)), np.eye(n)], [K, -gamma * np.eye(n)]])
        assert np.max(np.abs(A.T @ J + J @ A + gamma * J)) < 1e-14


def test_conformal_symplectic_generator_identity_needs_symmetric_K() -> None:
    """With a non-symmetric K the generator identity fails -- R1's hypothesis is real."""
    rng = np.random.default_rng(7)
    n, gamma = 8, 0.37
    J = symplectic_form(n)
    K = rng.standard_normal((n, n))  # deliberately not symmetrised
    A = np.block([[np.zeros((n, n)), np.eye(n)], [K, -gamma * np.eye(n)]])
    assert np.max(np.abs(A.T @ J + J @ A + gamma * J)) > 1e-2
