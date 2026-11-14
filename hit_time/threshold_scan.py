"""Re-derive the time-of-arrival threshold on a built event set: the threshold at which the toy's fraction of particle
hits with a time is HGCAL's (TARGET = 25.2 %, measured on the particle hits of 500 CMS tau events).

usage: threshold_scan.py <event set dir> [max events]

Every event of the hit-time builder holds, per hit, `rechit_time` (ns, -99 = no time) and `rechit_time_charge` (the
cell's in-time charge in fC). A hit has a time when that charge reaches the threshold, so the fraction for any other
threshold follows from the stored charges without a rebuild. Printed:
- the fraction of particle hits (truth_assignment >= 0) and of all hits with a time at the threshold the set was built
  with, and the energy fraction in timed particle hits;
- the consistency of the two columns: hits with a time whose charge is below the smallest charge of a timed hit;
- the fraction of particle hits with charge >= t for a grid of t, and the threshold that gives TARGET.
"""
import sys

import numpy as np
import ra_pickles

TARGET = 0.252
SENTINEL = -99.0

folder = sys.argv[1]
reader = ra_pickles.RandomAccessPicklesReader(folder)
n = reader.get_total() if len(sys.argv) < 3 else min(reader.get_total(), int(sys.argv[2]))
charge, timed, particle, energy = [], [], [], []
for i in range(n):
    ev = reader.get_element(i)
    charge.append(np.asarray(ev["rechit_time_charge"], dtype=np.float64))
    timed.append(np.asarray(ev["rechit_time"]) != SENTINEL)
    particle.append(np.asarray(ev["truth_assignment"]) >= 0)
    energy.append(np.asarray(ev["rechit_energy"], dtype=np.float64))
charge, timed, particle, energy = (np.concatenate(a) for a in (charge, timed, particle, energy))
print(f"{folder}: {n} events, {len(charge)} hits, {int(particle.sum())} particle hits ({particle.mean():.4f})")
print(f"built set: hits with a time {timed.mean():.4f} of all, {timed[particle].mean():.4f} of the particle hits; "
      f"energy in timed particle hits {energy[particle & timed].sum() / energy[particle].sum():.4f}; "
      f"noise hits with a time {int((timed & ~particle).sum())}")
built = charge[timed].min()
print(f"smallest in-time charge of a timed hit {built:.3f} fC; untimed hits at or above it: {int((~timed & (charge >= built)).sum())} "
      f"(hits above the threshold whose arrival falls outside the bunch window)")
q = charge[particle]
print("threshold [fC] -> fraction of particle hits with in-time charge >= threshold")
for t in (12, 20, 30, 35, 38, 39, 40, 41, 42, 45, 50, 60):
    print(f"  {t:5.1f}  {np.mean(q >= t):.4f}")
t_target = float(np.quantile(q, 1.0 - TARGET))
print(f"threshold for {TARGET:.3f} of the particle hits: {t_target:.2f} fC (fraction at 40 fC: {np.mean(q >= 40.0):.4f})")
