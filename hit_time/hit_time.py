"""Hit times of toy-calorimeter events, built the way CMSSW builds the HGCAL rec-hit time.

The generator of this branch (src/Calorimeter.cc, src/B4aEventAction.cc) stores, per simulation, every energy deposit in a sensor summed per
(particle, sensor, 1 ps slice): step_particle_id (index of the simulation's particles; -1 none), step_sensor_idx,
step_deposit (GeV) and step_time (ns since the collision). The event builder collects the deposits that make up an
event (capture_steps, the same selection the builder applies to the energies) and hit_times turns them into one time
per sensor.

Recipe (CMSSW_20_1_0_pre5: HGCDigitizer::accumulate, HGCFEElectronics::runShaperWithToT,
HGCalUncalibRecHitRecWeightsAlgo; constants of the 300 um silicon sensors, the toy's sensor thickness):
1. charge = deposit [keV] * KEV2FC.
2. every collision has its own time and its own position on the beam line (CMSSW: HLLHCEvtVtxGenerator draws a 4D
   vertex per generated collision, pileup collisions included): t_v ~ Gaussian(collision_sigma_ns), z_v ~
   Gaussian(collision_sigma_z_cm). The simulations are all made at the origin at t = 0, so the vertex enters as an
   arrival-time shift per particle: t_v + (|P - (0, 0, z_v)| - |P|) / c, with P the particle's first active impact
   point (the path-length change of a particle that reaches the same point from the displaced vertex; beta = 1; the
   shower itself is not re-simulated). A deposit attributed to no particle gets t_v. The particle simulations of one
   event are one collision; every pileup (minbias) simulation is its own collision. The defaults keep the total
   per-endcap spread at the value measured on the HGCAL tau sample (rms 0.131 ns of the per-endcap median of
   time - distance / c), split between time and z in the ratio of the HL-LHC values (0.18 ns : 4.5 cm): 0.102 ns and
   2.56 cm. Our HGCAL events fill one endcap and carry no vertex, so the split itself is not measured.
2b. out-of-time pileup (CMSSW Phase-2 mixing: bunch crossings -3..+3, 25 ns apart): the deposits of a collision of
   bunch crossing b are shifted by 25 b ns. Their energy enters the in-time hit with the HGCROC pulse-shape weight of
   the 25 ns bucket they fall in (HGCFEElectronics, adcPulse normalised to the in-time sample): bucket 0 (in time)
   1, bucket -1 (the previous crossing) 0.163 / 0.817, bucket -2 0.003 / 0.817, bucket +1 0.017 / 0.817, others 0
   (leak_energy). Deposits that fall in the in-time bucket also enter the time of arrival below. The in-time
   collisions' energies stay the builder's (all their deposits, whatever their time).
3. only deposits of the in-time bunch count for the time: 0 <= t + TOF_DELAY < BX.
4. time of arrival = the time at which the time-ordered cumulative charge of a cell crosses threshold_fc, linearly
   interpolated between the two deposits around the threshold; a cell at or below the threshold has no time.
   CMSSW's threshold is 12 fC; the toy's cells are larger, so the default is the threshold that gives HGCAL's
   fraction of timed hits (about 40 fC; threshold_scan.py re-derives it on a built set).
5. smearing: Gaussian of sigma = JITTER_NS * noise_fc / (in-time charge of the cell), plus one common Gaussian offset
   per event of sigma EVENT_OFFSET_NS (electronics).
6. valid only if 0 <= t + TOF_DELAY <= BX after smearing; stored in steps of TOA_LSB; hit time = that - TOF_DELAY =
   the arrival time since the nominal collision time, not corrected for the flight distance; no valid time -> SENTINEL.

Python 3.8 (the g4calo container) and numpy only.
"""
import numpy as np

KEV2FC = 0.044259            # fC per keV deposited in silicon (hgcalDigitizer_cfi)
TOF_DELAY = -9.0             # ns (CE-E): shifts the arrival into the 25 ns bunch window
BX = 25.0                    # ns
JITTER_NS = 5.0              # ns: time jitter at signal / noise = 1 (hgcROCParameters_cfi jitterNoise_ns)
EVENT_OFFSET_NS = 0.02       # ns (jitterConstant_ns): one common electronics offset per event
TOA_LSB = 0.0244             # ns (toaLSB_ns)
SENTINEL = -99.0             # "no valid time", as HGCAL's RecHitHGC_time
STEP_KEYS = ("step_particle_id", "step_sensor_idx", "step_deposit", "step_time")

C_CM_NS = 29.9792458
# HGCROC pulse shape per 25 ns sample (hgcROCParameters_cfi adcPulse), normalised to the in-time sample as in
# HGCFEElectronics: the weight with which charge deposited in bucket k appears in the in-time sample
ADC_PULSE = (0.00, 0.017, 0.817, 0.163, 0.003, 0.000)
BUCKET_WEIGHT = {0: 1.0, -1: ADC_PULSE[3] / ADC_PULSE[2], -2: ADC_PULSE[4] / ADC_PULSE[2], 1: ADC_PULSE[1] / ADC_PULSE[2]}

# CMSSW Phase-2 pileup scenario (Configuration/StandardSequences/python/Mixing.py, AVE_200_BX_25ns: 'B': (-3, 3)):
# the out-of-time bunch crossings mixed around the in-time one
CMS_OOT_BUNCHES = (-3, -2, -1, 1, 2, 3)

DEFAULTS = dict(threshold_fc=40.0, collision_sigma_ns=0.102, collision_sigma_z_cm=2.56, noise_gev=5e-6, jitter=True)


def charge_fc(deposit_gev):
    return np.asarray(deposit_gev, dtype=np.float64) * 1e6 * KEV2FC


def arrival_times(sensor, time, charge, threshold_fc):
    """Per cell with in-time deposits: (cell ids ascending, total in-time charge, threshold-crossing time or nan)."""
    win = (time + TOF_DELAY >= 0) & (time + TOF_DELAY < BX)
    s, t, q = sensor[win], time[win], charge[win]
    if not len(s):
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    order = np.lexsort((t, s))
    s, t, q = s[order], t[order], q[order]
    start = np.concatenate([[0], np.nonzero(np.diff(s))[0] + 1])
    cells = s[start]
    cum = np.cumsum(q)
    base = np.concatenate([[0.0], cum[start[1:] - 1]])
    group = np.repeat(np.arange(len(start)), np.diff(np.concatenate([start, [len(s)]])))
    cum = cum - base[group]
    total = np.add.reduceat(q, start)
    idx = np.where(cum > threshold_fc, np.arange(len(s)), len(s))
    k = np.minimum.reduceat(idx, start)
    fired = k < len(s)
    toa = np.full(len(start), np.nan)
    kf = k[fired]
    first = kf == start[fired]
    prev = np.maximum(kf - 1, 0)
    interp = (threshold_fc - cum[prev]) * (t[kf] - t[prev]) / np.maximum(cum[kf] - cum[prev], 1e-30) + t[prev]
    toa[fired] = np.where(first, t[kf], interp)
    return cells.astype(np.int64), total, toa


def kept_particles(phi, window):
    """The builder's pileup phase cut (EventGenerator._filter_simulations): a particle is kept if the azimuth of its
    first active impact, in [0, 2 pi), is inside the open window (start, end); the window may wrap around 2 pi."""
    phi = np.asarray(phi, dtype=np.float64)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    start, end = window
    if start > end:
        return np.logical_or(phi < end, phi > start)
    return np.logical_and(phi < end, phi > start)


def vertex_shift_ns(sim, t_v, z_v_cm):
    """Per particle of a simulation: the arrival-time shift of a collision at time t_v and beam-line position z_v
    (cm): t_v + (|P - (0, 0, z_v)| - |P|) / c, P = the first active impact point (stored in mm)."""
    x = np.asarray(sim["particles_first_active_impact_position_x"], dtype=np.float64) / 10.0
    y = np.asarray(sim["particles_first_active_impact_position_y"], dtype=np.float64) / 10.0
    z = np.asarray(sim["particles_first_active_impact_position_z"], dtype=np.float64) / 10.0
    return t_v + (np.sqrt(x**2 + y**2 + (z - z_v_cm) ** 2) - np.sqrt(x**2 + y**2 + z**2)) / C_CM_NS


def capture_steps(store, simulations, keeps, minbias, collision_sigma_ns, rng, collision_sigma_z_cm=0.0, bunch=0):
    """Append to `store` the (sensor, deposit, time) arrays of the deposits that enter the event.
    `bunch` b shifts the simulations' deposits by 25 b ns (out-of-time pileup).

    keeps None: every deposit of every simulation (the builder takes the energies of an uncut simulation from all its
    deposits, attributed to a particle or not). keeps = one boolean array per simulation (per particle): only the
    deposits of kept particles (the builder then takes the energies from the per-particle records; deposits attributed
    to no particle are dropped). minbias False: the simulations are one collision and share one time offset; True:
    one offset per simulation."""
    def vertex():
        t_v = rng.normal(0.0, collision_sigma_ns) if collision_sigma_ns > 0 else 0.0
        z_v = rng.normal(0.0, collision_sigma_z_cm) if collision_sigma_z_cm > 0 else 0.0
        return t_v, z_v

    shared = None if minbias else vertex()
    for k, sim in enumerate(simulations):
        missing = [key for key in STEP_KEYS if key not in sim]
        if missing:
            raise KeyError("simulation without hit-time keys %s: not made with the hit-time generator" % missing)
        p = np.asarray(sim["step_particle_id"], dtype=np.int64)
        s = np.asarray(sim["step_sensor_idx"], dtype=np.int64)
        e = np.asarray(sim["step_deposit"], dtype=np.float64)
        t = np.asarray(sim["step_time"], dtype=np.float64)
        t_v, z_v = shared if shared is not None else vertex()
        if z_v != 0.0 and p.size:
            shift = vertex_shift_ns(sim, t_v, z_v)
            if p.max() >= len(shift):
                raise ValueError("step_particle_id %d past the %d particles of the simulation" % (p.max(), len(shift)))
            offset = np.where(p >= 0, shift[np.clip(p, 0, None)] if len(shift) else t_v, t_v)
        else:
            offset = np.full(len(p), t_v)
        offset = offset + BX * bunch
        if keeps is not None:
            keep = np.asarray(keeps[k], dtype=bool)
            if p.size and p.max() >= len(keep):
                raise ValueError("step_particle_id %d past the %d particles of the simulation" % (p.max(), len(keep)))
            m = (p >= 0) & (keep[np.clip(p, 0, None)] if len(keep) else np.zeros(len(p), bool))
            s, e, t, offset = s[m], e[m], t[m], offset[m]
        store.append((s, e, t + offset))


def leak_energy(store, n_sensors):
    """The energy (GeV, raw) that out-of-time collisions add to the in-time hits: every deposit weighted by the pulse
    weight of its 25 ns bucket, floor((t + TOF_DELAY) / BX) (BUCKET_WEIGHT; 0 outside buckets -2..+1)."""
    out = np.zeros(n_sensors, dtype=np.float64)
    for s, e, t in store:
        bucket = np.floor((t + TOF_DELAY) / BX).astype(np.int64)
        w = np.zeros(len(t))
        for k, weight in BUCKET_WEIGHT.items():
            w[bucket == k] = weight
        np.add.at(out, s, e * w)
    return out


def hit_times(store, n_sensors, threshold_fc, noise_gev, rng, jitter=True):
    """One time and one in-time charge per sensor from the captured deposits: (time [ns] or SENTINEL, charge [fC])."""
    time_out = np.full(n_sensors, SENTINEL, dtype=np.float32)
    charge_out = np.zeros(n_sensors, dtype=np.float32)
    if not store:
        return time_out, charge_out
    sensor = np.concatenate([x[0] for x in store])
    charge = charge_fc(np.concatenate([x[1] for x in store]))
    time = np.concatenate([x[2] for x in store])
    cells, total, toa = arrival_times(sensor, time, charge, threshold_fc)
    if not len(cells):
        return time_out, charge_out
    if cells.max() >= n_sensors:
        raise ValueError("sensor index %d past the %d sensors" % (cells.max(), n_sensors))
    charge_out[cells] = total
    fired = np.isfinite(toa)
    shifted = toa + TOF_DELAY
    if jitter:
        sigma = JITTER_NS * charge_fc(noise_gev) / np.maximum(total, 1e-30)
        shifted = shifted + rng.normal(0.0, 1.0, len(cells)) * sigma + rng.normal(0.0, EVENT_OFFSET_NS)
    ok = fired & (shifted >= 0) & (shifted <= BX)
    stored = np.floor(shifted[ok] / TOA_LSB + 1e-9) * TOA_LSB - TOF_DELAY
    time_out[cells[ok]] = stored
    return time_out, charge_out
