"""Planted known-answer tests of hit_time.py (needs numpy only): python test_hit_time.py

Charges are given in fC through deposits = fC / (1e6 * KEV2FC) GeV. Threshold 12 fC in the first tests.
"""
import numpy as np

import hit_time as ht

GEV = 1.0 / (1e6 * ht.KEV2FC)          # GeV per fC


class Fixed:
    """A random source with planted values, to check what is drawn and how often."""
    def __init__(self, values):
        self.values, self.calls = list(values), 0

    def normal(self, mean, sigma, size=None):
        self.calls += 1
        if size is not None:
            return np.zeros(size)
        return mean + sigma * self.values.pop(0)


def sim(parts, sensors, fc, times):
    return {"step_particle_id": np.array(parts), "step_sensor_idx": np.array(sensors),
            "step_deposit": np.array(fc, dtype=np.float32) * GEV, "step_time": np.array(times, dtype=np.float32)}


def test_arrival_times():
    # cell 2: 5 fC at 11.0, 5 at 11.2, 4 at 11.4 (out of order) -> crosses 12 at 11.2 + (12 - 10) / 4 * 0.2 = 11.3
    # cell 5: one 20 fC deposit at 12 ns -> 12; cell 7: 3 fC in time + 100 fC at 40 ns (outside the window) -> none
    # cell 9: exactly 12 fC -> none (strictly above)
    sensor = np.array([2, 7, 2, 5, 2, 7, 9, 9])
    time = np.array([11.4, 11.0, 11.0, 12.0, 11.2, 40.0, 11.0, 11.5])
    charge = np.array([4.0, 3.0, 5.0, 20.0, 5.0, 100.0, 6.0, 6.0])
    cells, total, toa = ht.arrival_times(sensor, time, charge, 12.0)
    assert cells.tolist() == [2, 5, 7, 9] and np.allclose(total, [14.0, 20.0, 3.0, 12.0])
    assert np.allclose(toa[:2], [11.3, 12.0]) and np.isnan(toa[2]) and np.isnan(toa[3])
    # a higher threshold: cell 2 (14 fC) no longer fires, cell 5 (20 fC) does
    _c, _t, toa40 = ht.arrival_times(sensor, time, charge, 15.0)
    assert np.isnan(toa40[0]) and toa40[1] == 12.0


def test_kept_particles():
    phi = np.array([0.1, -0.1, 3.0, 6.2])               # -0.1 -> 6.183
    assert ht.kept_particles(phi, (0.0, 0.5)).tolist() == [True, False, False, False]
    assert ht.kept_particles(phi, (6.0, 0.2)).tolist() == [True, True, False, True]   # window wraps around 2 pi


def test_capture_and_times_two_showers_share_a_cell():
    # one collision of two particle simulations sharing cell 4: A 8 fC at 11.0 ns, B 8 fC at 11.5 ns -> neither alone
    # reaches 12 fC; together they cross at 11.0 + (12 - 8) / 8 * 0.5 = 11.25 ns. Cell 6 (A only, 30 fC at 11.1) -> 11.1.
    # A's deposit attributed to no particle (-1) in cell 4, 2 fC at 10.9, counts for an uncut simulation:
    # cumulative 2, 10, 18 -> crosses at 11.0 + (12 - 10) / 8 * 0.5 = 11.125 ns.
    a = sim([0, 0, -1], [4, 6, 4], [8.0, 30.0, 2.0], [11.0, 11.1, 10.9])
    b = sim([0], [4], [8.0], [11.5])
    store, rng = [], Fixed([0.0])
    ht.capture_steps(store, [a, b], None, False, 0.13, rng)
    assert rng.calls == 1                                # one collision offset for the whole particle set
    t, q = ht.hit_times(store, 10, 12.0, 5e-6, rng, jitter=False)
    lsb = lambda x: np.floor((x + ht.TOF_DELAY) / ht.TOA_LSB + 1e-9) * ht.TOA_LSB - ht.TOF_DELAY  # noqa: E731
    assert np.isclose(t[4], lsb(11.125), atol=1e-5) and np.isclose(t[6], lsb(11.1), atol=1e-5)
    assert np.isclose(q[4], 18.0, rtol=1e-6) and np.isclose(q[6], 30.0, rtol=1e-6)
    assert (t[[0, 1, 2, 3, 5, 7, 8, 9]] == ht.SENTINEL).all() and (q[[0, 1, 2, 3, 5, 7, 8, 9]] == 0).all()
    assert abs(t[6] - 11.1) <= ht.TOA_LSB


def test_collision_offset_shifts_and_pileup_draws_per_simulation():
    # the particle set is shifted by +1 sigma = 0.13 ns: cell 6 fires at 11.1 + 0.13
    a = sim([0], [6], [30.0], [11.1])
    store = []
    ht.capture_steps(store, [a], None, False, 0.13, Fixed([1.0]))
    t, _q = ht.hit_times(store, 10, 12.0, 5e-6, Fixed([]), jitter=False)
    assert abs(t[6] - 11.23) <= ht.TOA_LSB
    # two pileup simulations: one offset each (+1 and -2 sigma); particle 1 of the first is outside the phase window
    # and its deposit (cell 3) is dropped, as is a deposit attributed to no particle (cell 8)
    p1 = sim([0, 1, -1], [2, 3, 8], [30.0, 30.0, 30.0], [11.0, 11.0, 11.0])
    p2 = sim([0], [5], [30.0], [12.0])
    store, rng = [], Fixed([1.0, -2.0])
    ht.capture_steps(store, [p1, p2], [np.array([True, False]), np.array([True])], True, 0.1, rng)
    assert rng.calls == 2
    t, q = ht.hit_times(store, 10, 12.0, 5e-6, Fixed([]), jitter=False)
    assert abs(t[2] - 11.1) <= ht.TOA_LSB and abs(t[5] - 11.8) <= ht.TOA_LSB
    assert t[3] == ht.SENTINEL and t[8] == ht.SENTINEL and q[3] == 0 and q[8] == 0


def test_jitter_scale_and_window():
    # sigma = 5 ns * noise / charge: noise 5e-6 GeV = 0.2213 fC; a 22.13 fC cell has sigma 0.05 ns
    n = 20000
    s = sim([0] * n, list(range(n)), [22.13] * n, [12.0] * n)
    store = []
    ht.capture_steps(store, [s], None, False, 0.0, np.random.default_rng(1))
    t, _q = ht.hit_times(store, n, 12.0, 5e-6, np.random.default_rng(2), jitter=True)
    assert (t > 0).all()
    assert abs(np.std(t - np.mean(t)) - 0.05) < 0.003    # the common event offset moves the mean, not the spread
    # a cell that crosses the threshold after the bunch window has no time: 30 fC at 34.5 ns is outside [9, 34)
    late = sim([0], [1], [30.0], [34.5])
    store = []
    ht.capture_steps(store, [late], None, False, 0.0, np.random.default_rng(1))
    t, q = ht.hit_times(store, 3, 12.0, 5e-6, np.random.default_rng(2), jitter=False)
    assert t[1] == ht.SENTINEL and q[1] == 0


def test_vertex_position_shift():
    # a collision at z_v = +3 cm, t_v = 0: a particle whose first active impact is at (0, 96, 320) cm (stored in mm)
    # arrives earlier by (|P| - |P - (0,0,3)|) / c = (sqrt(96^2 + 320^2) - sqrt(96^2 + 317^2)) / c = 2.8724 cm / c
    # = 0.09581 ns; a deposit attributed to no particle is not shifted (t_v only)
    s = sim([0, -1], [6, 7], [50.0, 50.0], [12.0, 12.0])
    s.update(particles_first_active_impact_position_x=np.array([0.0]), particles_first_active_impact_position_y=np.array([960.0]),
             particles_first_active_impact_position_z=np.array([3200.0]))
    d = (np.sqrt(96.0**2 + 320.0**2) - np.sqrt(96.0**2 + 317.0**2)) / ht.C_CM_NS
    assert abs(d - 0.09581) < 2e-5
    assert np.allclose(ht.vertex_shift_ns(s, 0.25, 3.0), [0.25 - d])
    store, rng = [], Fixed([0.0, 1.0])                   # t_v = 0, z_v = +1 sigma = 3 cm
    ht.capture_steps(store, [s], None, False, 0.1, rng, collision_sigma_z_cm=3.0)
    assert rng.calls == 2
    assert np.allclose(store[0][2], [12.0 - d, 12.0], atol=1e-6)
    # on the other side of the collision point the same vertex arrives later: z_v = -3 cm -> + (|P + 3| - |P|) / c
    assert ht.vertex_shift_ns(s, 0.0, -3.0)[0] > 0


def test_out_of_time_bunch_and_leak():
    # a collision of the previous bunch crossing (bunch -1): a prompt 100 fC deposit at 12 ns lands at -13 ns, bucket
    # floor((-13 - 9) / 25) = -1 -> weight 0.163 / 0.817; a late one at 40 ns lands at 15 ns, bucket 0 -> weight 1 and
    # it enters the time of arrival (15 ns). Bunch +1: prompt at 12 -> 37 ns, bucket 1 -> 0.017 / 0.817.
    # Bunch -3: prompt at 12 -> -63 ns, bucket -3 -> 0.
    w1, wp = 0.163 / 0.817, 0.017 / 0.817
    prev = sim([0, 0], [1, 2], [100.0, 100.0], [12.0, 40.0])
    nxt = sim([0], [3], [100.0], [12.0])
    far = sim([0], [4], [100.0], [12.0])
    store = []
    ht.capture_steps(store, [prev], None, True, 0.0, Fixed([]), bunch=-1)
    ht.capture_steps(store, [nxt], None, True, 0.0, Fixed([]), bunch=1)
    ht.capture_steps(store, [far], None, True, 0.0, Fixed([]), bunch=-3)
    leak = ht.leak_energy(store, 6) / GEV
    assert np.allclose(leak, [0.0, 100.0 * w1, 100.0, 100.0 * wp, 0.0, 0.0], rtol=1e-5)
    t, q = ht.hit_times(store, 6, 12.0, 5e-6, Fixed([]), jitter=False)
    assert abs(t[2] - 15.0) <= ht.TOA_LSB and np.isclose(q[2], 100.0, rtol=1e-5)
    assert (t[[0, 1, 3, 4, 5]] == ht.SENTINEL).all() and (q[[0, 1, 3, 4, 5]] == 0).all()


def test_missing_keys_raise():
    try:
        ht.capture_steps([], [{"rechit_energy": np.zeros(1)}], None, False, 0.1, np.random.default_rng(0))
    except KeyError as e:
        assert "hit-time" in str(e)
    else:
        raise AssertionError("no error for a simulation without step keys")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("passed", name)
    print("ALL HIT-TIME TESTS PASSED")
