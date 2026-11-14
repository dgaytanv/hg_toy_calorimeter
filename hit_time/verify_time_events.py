"""Check the hit-time columns of one built event folder (build_events_time.py).

Per event: rechit_time and rechit_time_charge exist with one entry per hit; a hit with a time has an in-time charge
above the threshold (hard); the in-time charge is not more than the hit's raw energy (rechit_energy / sensors_scaling,
which also holds noise and out-of-time deposits) (hard); valid times lie in the bunch window [9, 34] ns (hard).
Reported: the fraction of hits and of energy with a time, hits above threshold without a time (smeared out of the
window), time - distance / c. Exit status 1 on a hard failure.
usage: verify_time_events.py <ra_pickles event dir> <calibration.bin> <threshold fC>
"""
import gzip
import pickle
import sys

import numpy as np
import ra_pickles

import hit_time as ht

folder, calib_path, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
with gzip.open(calib_path, "rb") as f:
    scaling = np.asarray(pickle.load(f)["sensors_scaling"], dtype=np.float64)
reader = ra_pickles.RandomAccessPicklesReader(folder)
n = reader.get_total()
bad = []
tot = dict(hits=0, timed=0, e=0.0, e_timed=0.0, above_no_time=0, above=0)
delta, ratio = [], []
for i in range(n):
    ev = reader.get_element(i)
    if "rechit_time" not in ev or "rechit_time_charge" not in ev:
        bad.append((i, "no rechit_time / rechit_time_charge")); continue
    t = np.asarray(ev["rechit_time"], dtype=np.float64)
    q = np.asarray(ev["rechit_time_charge"], dtype=np.float64)
    e = np.asarray(ev["rechit_energy"], dtype=np.float64)
    idx = np.asarray(ev["rechit_sensor_idx"], dtype=np.int64)
    if not (len(t) == len(q) == len(e) == len(idx)):
        bad.append((i, "lengths %d %d %d %d" % (len(t), len(q), len(e), len(idx)))); continue
    valid = t > -0.5
    if (~valid & (t != ht.SENTINEL)).any():
        bad.append((i, "invalid hits without the sentinel"))
    # The charge is stored in float32, the builder compares in float64. A cell whose
    # charge is above the threshold by less than the float32 spacing is stored as exactly the threshold and has a time.
    # So a timed hit must not be BELOW the float32 threshold; equality is the rounding case, counted and printed.
    if (q[valid] < np.float64(np.float32(thr))).any():
        bad.append((i, "%d timed hits below the threshold" % (q[valid] < np.float64(np.float32(thr))).sum()))
    tot["at_threshold"] = tot.get("at_threshold", 0) + int((q[valid] == np.float64(np.float32(thr))).sum())
    raw_fc = ht.charge_fc(e / scaling[idx])
    if "rechit_oot_energy" in ev:                        # out-of-time pileup: part of the hit energy, never more than it
        oot = np.asarray(ev["rechit_oot_energy"], dtype=np.float64)
        if len(oot) != len(e) or (oot < 0).any() or (oot > e * (1 + 1e-4) + 1e-9).any():
            bad.append((i, "rechit_oot_energy inconsistent with rechit_energy"))
        tot["oot"] = tot.get("oot", 0.0) + oot.sum(); tot["oot_hits"] = tot.get("oot_hits", 0) + int((oot > 0).sum())
    if (q > raw_fc * (1 + 1e-4) + 1e-3).any():
        bad.append((i, "in-time charge above the hit's raw energy in %d hits" % (q > raw_fc * (1 + 1e-4) + 1e-3).sum()))
    if valid.any() and (t[valid].min() < 9.0 - ht.TOA_LSB or t[valid].max() > 34.0):
        bad.append((i, "valid time outside [9, 34]: %.3f .. %.3f" % (t[valid].min(), t[valid].max())))
    above = q > thr
    tot["hits"] += len(t); tot["timed"] += int(valid.sum()); tot["e"] += e.sum(); tot["e_timed"] += e[valid].sum()
    tot["above"] += int(above.sum()); tot["above_no_time"] += int((above & ~valid).sum())
    d = np.sqrt(np.asarray(ev["rechit_x"], dtype=np.float64) ** 2 + np.asarray(ev["rechit_y"], dtype=np.float64) ** 2
                + np.asarray(ev["rechit_z"], dtype=np.float64) ** 2) / 29.9792458
    delta.append(t[valid] - d[valid]); ratio.append(q[above] / raw_fc[above])
print("folder: %s, %d events, threshold %.1f fC" % (folder, n, thr))
if tot["hits"]:
    print("hits %d, with a time %.4f; energy with a time %.4f" % (tot["hits"], tot["timed"] / tot["hits"], tot["e_timed"] / max(tot["e"], 1e-30)))
    print("hits above threshold without a time: %d of %d" % (tot["above_no_time"], tot["above"]))
    print("timed hits whose stored charge equals the threshold (float32 rounding): %d" % tot.get("at_threshold", 0))
    if "oot" in tot:
        print("out-of-time pileup: %.5f of the energy, in %.4f of the hits" % (tot["oot"] / max(tot["e"], 1e-30), tot["oot_hits"] / tot["hits"]))
delta = np.concatenate(delta) if delta else np.zeros(0)
ratio = np.concatenate(ratio) if ratio else np.zeros(0)
if len(delta):
    print("time - distance / c [ns], percentiles (1, 16, 50, 84, 99):", np.round(np.percentile(delta, [1, 16, 50, 84, 99]), 4).tolist())
    print("in-time charge / raw hit charge of hits above threshold, percentiles (1, 50): %s" % np.round(np.percentile(ratio, [1, 50]), 4).tolist())
for b in bad[:10]:
    print("BAD event %d: %s" % b)
ok = not bad and n > 0
print("VERIFY TIME EVENTS %s" % ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
