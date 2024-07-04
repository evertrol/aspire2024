import numpy as np

from aspire.integrators.states import OrbitState


def test_orbitstate():
    state = OrbitState(0, 1, -1, 0)
    assert state.x == 0
    assert state.y == 1
    assert state.u == -1
    assert state.v == 0

    state = OrbitState(np.inf, np.nan, -1j, 0)
    assert state.x == np.inf
    assert np.isnan(state.y)
    assert state.u == -1j
    assert state.v == 0
