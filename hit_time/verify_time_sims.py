"""Check the hit-time keys of one ra_pickles simulation folder (simulated with the generator of this branch).

Per simulation, against keys the unpatched generator already writes:
1. the step deposits summed per (particle, sensor), particle >= 0, equal hit_particle_deposit of that pair;
2. the step deposits summed per sensor (all particles, -1 included) equal rechit_energy of that sensor;
3. step_particle_id is in [-1, number of particles); step_time is finite and not before the earliest possible arrival
   at the calorimeter front (320 cm / c = 10.67 ns), for every sim type (the face gun starts at its flight time).
Exit status 1 on any failure. usage: verify_time_sims.py <ra_pickles dir> <simtype>
"""
import sys

import numpy as np
import ra_pickles

folder, simtype = sys.argv[1], sys.argv[2]
T_MIN = 320.0 / 29.9792458 - 0.01
reader = ra_pickles.RandomAccessPicklesReader(folder)
n = reader.get_total()
worst_pair = worst_cell = 0.0
bad = []
slices, raw, unattributed, t_lo, t_hi, late = [], [], [], [], [], []
for i in range(n):
    d = reader.get_element(i)
    sim = d[0] if isinstance(d, tuple) else d
    p = np.asarray(sim["step_particle_id"], dtype=np.int64)
    s = np.asarray(sim["step_sensor_idx"], dtype=np.int64)
    e = np.asarray(sim["step_deposit"], dtype=np.float64)
    t = np.asarray(sim["step_time"], dtype=np.float64)
    if not (len(p) == len(s) == len(e) == len(t)):
        bad.append((i, "lengths differ")); continue
    slices.append(len(p)); raw.append(int(sim["step_count_raw"]))
    if not len(p):
        continue
    n_part = len(sim["particles_pdgid"])
    if p.min() < -1 or p.max() >= n_part:
        bad.append((i, "particle id out of range [%d, %d] for %d particles" % (p.min(), p.max(), n_part)))
    if not np.isfinite(t).all() or t.min() < T_MIN:
        bad.append((i, "time min %.4f ns (limit %.4f)" % (t.min(), T_MIN)))
    t_lo.append(t.min()); t_hi.append(np.percentile(t, 99)); late.append(e[t >= 34.0].sum() / e.sum())
    unattributed.append(e[p < 0].sum() / e.sum())
    # 1. per (particle, sensor)
    hp = np.asarray(sim["hit_particle_id"], dtype=np.int64)
    hs = np.asarray(sim["hit_particle_sensor_idx"], dtype=np.int64)
    hd = np.asarray(sim["hit_particle_deposit"], dtype=np.float64)
    big = 1 << 32
    att = p >= 0
    keys, inv = np.unique(p[att] * big + s[att], return_inverse=True)
    sums = np.bincount(inv, weights=e[att], minlength=len(keys))
    hkeys = hp * big + hs
    order = np.argsort(hkeys)
    if len(keys) != len(hkeys) or not np.array_equal(keys, hkeys[order]):
        bad.append((i, "(particle, sensor) pairs differ: %d step pairs, %d hit records" % (len(keys), len(hkeys))))
    else:
        worst_pair = max(worst_pair, float(np.max(np.abs(sums - hd[order]) / np.maximum(hd[order], 1e-12))))
    # 2. per sensor
    ridx = np.asarray(sim["rechit_idx"], dtype=np.int64)
    re = np.asarray(sim["rechit_energy"], dtype=np.float64)
    cells, cinv = np.unique(s, return_inverse=True)
    csum = np.bincount(cinv, weights=e, minlength=len(cells))
    rorder = np.argsort(ridx)
    if len(cells) != len(ridx) or not np.array_equal(cells, ridx[rorder]):
        bad.append((i, "sensors differ: %d with steps, %d rechits" % (len(cells), len(ridx))))
    else:
        worst_cell = max(worst_cell, float(np.max(np.abs(csum - re[rorder]) / np.maximum(re[rorder], 1e-12))))
print("folder: %s (%s), %d simulations" % (folder, simtype, n))
if slices:
    print("slices/sim: mean %.0f max %d; raw steps/sim: mean %.0f; slices per raw step %.3f"
          % (np.mean(slices), np.max(slices), np.mean(raw), np.sum(slices) / max(np.sum(raw), 1)))
if t_lo:
    print("step time [ns]: min %.3f, 99th percentile (median over sims) %.2f; deposit later than 34 ns: mean %.4f max %.4f"
          % (np.min(t_lo), np.median(t_hi), np.mean(late), np.max(late)))
    print("deposit attributed to no particle: mean %.4f max %.4f" % (np.mean(unattributed), np.max(unattributed)))
print("worst relative difference: (particle, sensor) sums vs hit_particle_deposit %.2e; sensor sums vs rechit_energy %.2e"
      % (worst_pair, worst_cell))
for b in bad[:10]:
    print("BAD sim %d: %s" % b)
ok = not bad and worst_pair < 1e-4 and worst_cell < 1e-4
print("VERIFY TIME %s" % ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
