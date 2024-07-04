import numpy as np

from aspire.integrators.states import OrbitState
from aspire.integrators.core import integrate_euler


def test_integrate_euler():
    state = OrbitState(0, 0, 0, 0)
    history, times = integrate_euler(state, 0.1)
    np.testing.assert_allclose(history[:11], np.arange(0, 1.1, 0.1))
    x = [time.x for time in times]
    y = [time.y for time in times]
    u = [time.u for time in times]
    v = [time.v for time in times]
    assert np.array_equal(x, [0, 0] + 10 * [np.nan], equal_nan=True)
    assert np.array_equal(y, [0, 0] + 10 * [np.nan], equal_nan=True)
    assert np.array_equal(u, [0] + 11 * [np.nan], equal_nan=True)
    assert np.array_equal(v, [0] + 11 * [np.nan], equal_nan=True)
