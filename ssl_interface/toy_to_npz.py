#!/usr/bin/env python3
"""Convert the toy-calorimeter NanoML parquet (one row per event) to the AI-ready npz schema of arXiv:2606.07667.

One compressed npz per parquet file, holding the paper's three arrays and one clearly separated toy extra:
  measurements  Table III fields, one row per hit (the toy is single-sample: sample = 0, subsystem = 0)
  labels        Table IV fields, row-aligned to measurements, N = 3 contributor slots of which only con0 is filled
  event_label   Table V fields, one row per SimCluster (the particles with >= 1 rechit)
  toy_extra     toy-only per-hit fields (layer, area, sensor), row-aligned to measurements; NOT part of the schema
Field meanings, units and sentinels are in FIELD_DOCS and are written to <out-dir>/metadata.md in the style of the
paper's Appendix A. Positions follow EVAL_PROTOCOL section 2: mm = float32(float64(cm) * 10.0).

Field names (--naming, required): `bic` writes the names and integer types of the ePIC BIC files (amplitude,
Deposit_energy, con_index0 .. con_parentEnergy2, Energy; event_label particle, charge, parentnumbers and parentindex
as int32), the naming the owner chose as canonical on 2026-10-30 (REPORT_B, Q1); `paper` writes the names of the
paper's Tables III-V, as the first conversion (job 61737, data/npz/pufree) did. build() and validate() work in the
paper names; the arrays are renamed (and the four event_label integers widened) only for writing.

Each file is written under a .partial name, re-opened and checked against a second read of its parquet file that
uses a different pyarrow code path (pyarrow.compute kernels over the chunked columns), and renamed only if every
check passes: E, x, y, z bit-for-bit, key columns of measurements and labels identical, n_contributions sum equal to
the number of matched hits, every label field against its definition, the join labels.con0_particle -> event_label,
and the per-particle sum of labels.deposit_energy against SimCluster_truthAssistedEnergy (relative difference
reported). A summary goes to <out-dir>/conversion_summary.json. The SHA256SUMS list is written by the calling job
(jobs/convert.sbatch) as its last step, so a downstream afterok covers the data and its hashes.
"""

import argparse
import hashlib
import json
import os
import re
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

N_CONTRIB = 3
EVENTS_PER_FILE = 1000  # global event index = file_index * EVENTS_PER_FILE + row of the event in its file
UINT16_MAX = np.iinfo(np.uint16).max
# PDG 2024 masses (GeV) by |pid|, and charges (e) by signed pid, of the species present in the toy sets
PDG_MASS_GEV = {11: 0.00051099895, 13: 0.1056583755, 211: 0.13957039, 22: 0.0}
PDG_CHARGE = {11: -1, -11: 1, 13: -1, -13: 1, 211: 1, -211: -1, 22: 0}

KEYS = ("event", "subsystem", "hit", "sample")
MEASUREMENTS_DTYPE = np.dtype([
    ("event", "<u8"), ("subsystem", "<u2"), ("hit", "<u2"), ("sample", "<u2"),
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("deposit_energy", "<f4"), ("time", "<f4"),
    ("x_unc", "<f4"), ("y_unc", "<f4"), ("z_unc", "<f4"), ("deposit_energy_unc", "<f4"), ("time_unc", "<f4"),
])
CONTRIB_FIELDS = (
    ("particle", "<u2"), ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("deposit_energy", "<f4"), ("time", "<f4"),
    ("px", "<f4"), ("py", "<f4"), ("pz", "<f4"), ("energy", "<f4"), ("pid", "<i4"), ("parent_pid", "<i4"),
    ("parent_energy", "<f4"),
)
LABELS_DTYPE = np.dtype([
    ("event", "<u8"), ("subsystem", "<u2"), ("hit", "<u2"), ("sample", "<u2"),
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("deposit_energy", "<f4"), ("time", "<f4"), ("n_contributions", "<u2"),
] + [(f"con{n}_{name}", kind) for n in range(N_CONTRIB) for name, kind in CONTRIB_FIELDS])
EVENT_LABEL_DTYPE = np.dtype([
    ("event", "<u8"), ("particle", "<u2"), ("Vx", "<f4"), ("Vy", "<f4"), ("Vz", "<f4"),
    ("px", "<f4"), ("py", "<f4"), ("pz", "<f4"), ("energy", "<f4"), ("time", "<f4"), ("charge", "i1"),
    ("pid", "<i4"), ("parentnumbers", "u1"), ("parentpid", "<i4"), ("parentindex", "<u2"),
])
TOY_EXTRA_DTYPE = np.dtype([("event", "<u8"), ("hit", "<u2"), ("layer", "u1"), ("area", "<f4"), ("sensor", "<i4")])
ARRAYS = {"measurements": MEASUREMENTS_DTYPE, "labels": LABELS_DTYPE, "event_label": EVENT_LABEL_DTYPE,
          "toy_extra": TOY_EXTRA_DTYPE}

# --naming bic: the field names and integer types of the ePIC BIC files (fields not listed keep name and type)
NAMINGS = ("bic", "paper")
CONTRIB_BIC_NAMES = ("index", "x", "y", "z", "DepositEnergy", "time", "px", "py", "pz", "Energy", "pid", "parentpid",
                     "parentEnergy")   # con_<name><n> in the BIC files, in CONTRIB_FIELDS order
BIC_NAMES = {
    "measurements": {"deposit_energy": "amplitude", "deposit_energy_unc": "amplitude_unc"},
    "labels": {"deposit_energy": "Deposit_energy",
               **{f"con{n}_{paper}": f"con_{bic}{n}" for n in range(N_CONTRIB)
                  for (paper, _), bic in zip(CONTRIB_FIELDS, CONTRIB_BIC_NAMES)}},
    "event_label": {"energy": "Energy"},
}
BIC_TYPES = {"event_label": {"particle": "<i4", "charge": "<i4", "parentnumbers": "<i4", "parentindex": "<i4"}}

RECHIT_BRANCHES = {"x": "RecHitHGC_x", "y": "RecHitHGC_y", "z": "RecHitHGC_z", "energy": "RecHitHGC_energy",
                   "idx": "RecHitHGC_SimClusterBestMatchIdx", "layer": "RecHitHGC_layer", "area": "RecHitHGC_area",
                   "sensor": "RecHitHGC_detId"}
SIMCLUSTER_BRANCHES = {"ipx": "SimCluster_impactPoint_x", "ipy": "SimCluster_impactPoint_y",
                       "ipz": "SimCluster_impactPoint_z", "T": "SimCluster_boundaryEnergy",
                       "pid": "SimCluster_pdgId", "tae": "SimCluster_truthAssistedEnergy",
                       "merged": "SimCluster_merged"}

NOT_ASSIGNED = "NOT ASSIGNED in the toy: 0 here means 'no value', not a measured zero."
SLOT_NOTE = ("Sentinel 0 for empty slots, which collides with particle 0: always mask by n_contributions (only "
             "slots n < n_contributions are valid).")
KINEMATICS = ("Derived, not simulated: E = T + m(|pid|) with T = SimCluster_boundaryEnergy (the TRUE KINETIC energy, "
              "see the converter docstring of toy_to_parquet.py) and the PDG masses e 0.00051099895, mu 0.1056583755, "
              "pi+- 0.13957039, gamma 0 GeV; |p| = sqrt(T (T + 2m)); direction = unit vector from the origin to the "
              "front-face crossing (con0_x, con0_y, con0_z). Approximation: the toy's particles start at the "
              "interaction point and fly straight (no magnetic field); the vertex itself is not in the parquet. "
              "For a merged SimCluster (several close-by particles merged by the event builder) T is the summed "
              "kinetic energy and m the mass of the stored pid. Recover the kinetic energy as energy - m(|pid|).")
FIELD_DOCS = {
    "measurements": {
        "event": "`uint64` integer index, dimensionless. One toy event (one PU-free bunch crossing: ~74 particles "
                 "shot into one endcap). event = file_index x 1000 + row of the event in its parquet file "
                 "(pufree_00k.parquet has file_index k); the converter checks it against the parquet event_id "
                 "(see the scheme line above). Rows are sorted by event, and within an event hits keep the parquet "
                 "row order. Grouping key for batching, not a learnable feature.",
        "subsystem": "`uint16` categorical index, dimensionless. 0 = the toy silicon calorimeter, the only subsystem "
                     "of this set (the toy's specialisation of the general `detector` field; in the BIC files 0 "
                     "means AstroPix, so mixing the two needs a shared subsystem registry). Not a learnable "
                     "feature.",
        "hit": "`uint16` integer index, dimensionless. Sequential index of the hit (one sensor above threshold) "
               "within its event, 0 .. n_hits - 1 in parquet row order. (event, subsystem, hit) identifies a hit.",
        "sample": "`uint16` integer index, dimensionless. Always 0: the toy is single-sample (one energy per "
                  "sensor, no waveform and no TOA row).",
        "x": "`float32` scalar, unit: mm. x of the centre of the fired sensor (RecHitHGC_x in cm, stored as "
             "float32(float64(cm) x 10)). Laboratory frame, z along the beam, nominal interaction point at "
             "the origin.",
        "y": "`float32` scalar, unit: mm. y of the sensor centre, as x.",
        "z": "`float32` scalar, unit: mm. z of the sensor centre, as x. The toy covers one endcap: front face at "
             "z = 3200 mm, z up to ~5090 mm.",
        "deposit_energy": "`float32` scalar, unit: GeV. Calibrated hit energy (RecHitHGC_energy), the toy's "
                          "measured amplitude; spans ~1e-5 to ~1e2 GeV, logarithmic scaling recommended. Truth "
                          "counterpart: deposit_energy in labels. The current BIC files name this field "
                          "`amplitude`.",
        "time": "`float32` scalar, unit: ns. " + NOT_ASSIGNED + " The toy has no timing (the parquet carries the "
                "sentinel -99, which is not copied).",
        "x_unc": "`float32` scalar, unit: mm. One-sigma uncertainty on x for a position uniform within the cell: "
                 "sqrt(area) / sqrt(12) with area = RecHitHGC_area (cm^2, x 100 to mm^2), i.e. a SQUARE-CELL "
                 "ASSUMPTION (side = sqrt(area)); the cell size encodes the granularity, which in the toy follows "
                 "the layer and radius.",
        "y_unc": "`float32` scalar, unit: mm. Same value as x_unc (square-cell assumption).",
        "z_unc": "`float32` scalar, unit: mm. " + NOT_ASSIGNED + " The sensor depth is not in the parquet.",
        "deposit_energy_unc": "`float32` scalar, unit: GeV. " + NOT_ASSIGNED + " The current BIC files name this "
                              "field `amplitude_unc`.",
        "time_unc": "`float32` scalar, unit: ns. " + NOT_ASSIGNED,
    },
    "labels": {
        "event": "`uint64` integer index. Join key, identical row by row to event in measurements (the arrays are "
                 "row-aligned). Not a learnable feature.",
        "subsystem": "`uint16` categorical index. Join key, identical to measurements (always 0).",
        "hit": "`uint16` integer index. Join key, identical to measurements.",
        "sample": "`uint16` integer index. Join key, identical to measurements (always 0).",
        "x": "`float32` scalar, unit: mm. NOT A TRUTH POSITION: the toy stores no per-hit truth position, so this "
             "is the sensor centre, a copy of x in measurements.",
        "y": "`float32` scalar, unit: mm. Sensor centre, as x.",
        "z": "`float32` scalar, unit: mm. Sensor centre, as x.",
        "deposit_energy": "`float32` scalar, unit: GeV. Energy attributed to particles in this hit: the calibrated "
                          "hit energy for a hit matched to a SimCluster (n_contributions = 1), 0 for a noise hit "
                          "(n_contributions = 0). The toy keeps only the largest depositor of each sensor, so the "
                          "whole hit energy is attributed to con0.",
        "time": "`float32` scalar, unit: ns. " + NOT_ASSIGNED,
        "n_contributions": "`uint16` integer count. 1 if the hit is matched to a SimCluster "
                           "(RecHitHGC_SimClusterBestMatchIdx >= 0), 0 for a noise hit. The toy records only the "
                           "largest depositor per sensor, so values above 1 never occur: a lower bound on the "
                           "true multiplicity, not an overlap measure.",
        "con0_particle": "`uint16` integer index. Row of the matched SimCluster within its event "
                         "(RecHitHGC_SimClusterBestMatchIdx), the foreign key into particle of event_label (same "
                         "event). " + SLOT_NOTE,
        "con0_x": "`float32` scalar, unit: mm. x where the particle crosses the calorimeter front face (z = 3200 "
                  "mm) on the straight line from its vertex along its momentum (SimCluster_impactPoint_x x 10). "
                  "The same for every hit of the particle: NOT the interaction position in this sensor (the toy "
                  "has none). " + SLOT_NOTE,
        "con0_y": "`float32` scalar, unit: mm. y of the front-face crossing, as con0_x.",
        "con0_z": "`float32` scalar, unit: mm. z of the front-face crossing (3200 mm), as con0_x.",
        "con0_deposit_energy": "`float32` scalar, unit: GeV. Equal to deposit_energy (the single stored "
                               "contributor carries the whole hit). " + SLOT_NOTE,
        "con0_time": "`float32` scalar, unit: ns. " + NOT_ASSIGNED,
        "con0_px": "`float32` scalar, unit: GeV/c. Momentum x of the matched particle at production. " + KINEMATICS
                   + " " + SLOT_NOTE,
        "con0_py": "`float32` scalar, unit: GeV/c. Momentum y, as con0_px.",
        "con0_pz": "`float32` scalar, unit: GeV/c. Momentum z, as con0_px.",
        "con0_energy": "`float32` scalar, unit: GeV. Total energy E = T + m(|pid|) of the matched particle at "
                       "production (see con0_px). " + SLOT_NOTE,
        "con0_pid": "`int32` categorical identifier. PDG ID of the matched particle (SimCluster_pdgId): 22, +-11, "
                    "+-13, +-211 in the toy. Sentinel 0 for noise hits.",
        "con0_parent_pid": "`int32` categorical identifier. 0: the toy keeps no ancestry; every SimCluster is a "
                           "primary particle.",
        "con0_parent_energy": "`float32` scalar, unit: GeV. 0: no ancestry in the toy.",
    },
    "event_label": {
        "event": "`uint64` integer index. Join key, the same event index as in measurements and labels.",
        "particle": "`uint16` integer index. Row of the SimCluster within its event, 0 .. n - 1: the key that "
                    "con{n}_particle in labels points to. One row per SimCluster = one per particle with >= 1 "
                    "rechit (particles without a rechit are not stored, as in the parquet).",
        "Vx": "`float32` scalar, unit: mm. " + NOT_ASSIGNED + " The vertex is not in the parquet; the toy's "
              "particles start at the interaction point.",
        "Vy": "`float32` scalar, unit: mm. " + NOT_ASSIGNED,
        "Vz": "`float32` scalar, unit: mm. " + NOT_ASSIGNED,
        "px": "`float32` scalar, unit: GeV/c. Momentum x at production. " + KINEMATICS,
        "py": "`float32` scalar, unit: GeV/c. Momentum y at production (see px).",
        "pz": "`float32` scalar, unit: GeV/c. Momentum z at production (see px).",
        "energy": "`float32` scalar, unit: GeV. Total energy E = T + m(|pid|) at production (see px).",
        "time": "`float32` scalar, unit: ns. " + NOT_ASSIGNED,
        "charge": "`int8` integer scalar, unit: e. From the pid: e- -1, e+ +1, mu- -1, mu+ +1, pi+ +1, pi- -1, "
                  "gamma 0.",
        "pid": "`int32` categorical identifier. PDG ID (SimCluster_pdgId).",
        "parentnumbers": "`uint8` integer count. 0: every SimCluster is a primary particle (no ancestry in the "
                         "toy).",
        "parentpid": "`int32` categorical identifier. 0 (primary, no parent).",
        "parentindex": "`uint16` integer index. 0 (primary, no parent).",
    },
    "toy_extra": {
        "event": "`uint64` integer index. Join key, identical row by row to measurements.",
        "hit": "`uint16` integer index. Join key, identical row by row to measurements.",
        "layer": "`uint8` integer index. Active layer of the sensor, 0..55 (RecHitHGC_layer). Toy-only; a model "
                 "input only in a configuration labelled C1+.",
        "area": "`float32` scalar, unit: cm^2. Sensor area (RecHitHGC_area); x_unc and y_unc are derived from it.",
        "sensor": "`int32` integer index. Sensor index in the toy geometry (RecHitHGC_detId). Not a learnable "
                  "feature.",
    },
}
FIELD_DOCS["labels"].update({
    f"con{n}_{name}": f"`{np.dtype(kind).name}`. Contributor slot {n}: same definition as con0_{name}. Never filled "
                      "in the toy (always 0), which keeps one particle per hit. " + SLOT_NOTE
    for n in range(1, N_CONTRIB) for name, kind in CONTRIB_FIELDS})
SLOT_TOKEN = re.compile(r"\bcon(\d+)_([a-z_]+)\b")   # a paper slot name in FIELD_DOCS text, e.g. con0_px
BIC_DOC_EDITS = (   # exact edits of the FIELD_DOCS text for --naming bic; each must occur (write_metadata checks)
    (" The current BIC files name this field `amplitude`.", ""),
    (" The current BIC files name this field `amplitude_unc`.", ""),
    ("Truth counterpart: deposit_energy in labels.", "Truth counterpart: Deposit_energy in labels."),
    ("Equal to deposit_energy", "Equal to Deposit_energy"),
    ("attributed to con0.", "attributed to contributor slot 0."),
    ("Recover the kinetic energy as energy - m(|pid|).", "Recover the kinetic energy as Energy - m(|pid|)."),
    ("con{n}_particle", "con_index{n}"),
)


def named_dtype(array: str, naming: str) -> np.dtype:
    """dtype of `array` as written under `naming`: the same field order, BIC names and integer types for "bic"."""
    base = ARRAYS[array]
    if naming == "paper":
        return base
    names, types = BIC_NAMES.get(array, {}), BIC_TYPES.get(array, {})
    return np.dtype([(names.get(f, f), types.get(f, base.fields[f][0].str)) for f in base.names])


def to_naming(arr: np.ndarray, array: str, naming: str) -> np.ndarray:
    """`arr` (paper names) under `naming`: a view if only names change, else a field-by-field copy of the values."""
    dt = named_dtype(array, naming)
    if [dt.fields[n][0] for n in dt.names] == [arr.dtype.fields[n][0] for n in arr.dtype.names]:
        return arr.view(dt)
    out = np.empty(len(arr), dtype=dt)
    for src, dst in zip(arr.dtype.names, dt.names):
        out[dst] = arr[src]
        if dt.fields[dst][0] != arr.dtype.fields[src][0] and not np.array_equal(out[dst], arr[src]):
            raise ValueError(f"{array}.{src} changes value as {dt.fields[dst][0]}")
    return out


def paper_view(arr: np.ndarray, array: str) -> np.ndarray:
    """View of `array` read from a file of either naming under the paper's field names (types as stored)."""
    base, fields = ARRAYS[array], arr.dtype.fields
    if len(arr.dtype.names) != len(base.names):
        raise ValueError(f"{array}: {len(arr.dtype.names)} fields, the paper dtype has {len(base.names)}")
    return arr.view(np.dtype({"names": list(base.names), "formats": [fields[n][0] for n in arr.dtype.names],
                              "offsets": [fields[n][1] for n in arr.dtype.names], "itemsize": arr.dtype.itemsize}))


def field_docs(array: str, naming: str) -> dict:
    """FIELD_DOCS of `array` keyed and worded for `naming` (BIC: names in the text translated, the dtype word taken
    from the written dtype, a widened integer and the paper's name of a renamed field stated)."""
    if naming == "paper":
        return FIELD_DOCS[array]
    slot = {paper: bic for (paper, _), bic in zip(CONTRIB_FIELDS, CONTRIB_BIC_NAMES)}
    base, dt = ARRAYS[array], named_dtype(array, naming)
    out = {}
    for paper_name, name in zip(base.names, dt.names):
        text = FIELD_DOCS[array][paper_name]
        for old, new in BIC_DOC_EDITS:
            text = text.replace(old, new)
        text = SLOT_TOKEN.sub(lambda m: f"con_{slot[m.group(2)]}{m.group(1)}" if m.group(2) in slot else m.group(0),
                              text)
        kind, paper_kind = dt.fields[name][0], base.fields[paper_name][0]
        text = re.sub(r"^`[a-z0-9]+`", f"`{kind.name}`", text)
        if kind != paper_kind:
            text += f" Stored as {kind.name} as in the BIC files ({paper_kind.name} in the paper-named files)."
        if name != paper_name:
            text += f" Paper (Tables III-V) name: `{paper_name}`."
        out[name] = text
    return out


def file_index_of(path: Path) -> int:
    m = re.fullmatch(r".*_(\d{3})\.parquet", path.name)
    if not m:
        raise ValueError(f"{path.name}: expected <set>_NNN.parquet")
    return int(m.group(1))


def mm(cm_values: np.ndarray) -> np.ndarray:
    """EVAL_PROTOCOL section 2: x_mm = np.float32(np.float64(x_cm) * 10.0)."""
    return (cm_values.astype(np.float64) * 10.0).astype(np.float32)


def lookup(table: dict, keys: np.ndarray, what: str) -> np.ndarray:
    out = np.full(keys.shape, np.nan)
    for k, v in table.items():
        out[keys == k] = v
    if np.isnan(out).any():
        raise ValueError(f"no {what} for pid(s) {sorted(set(keys[np.isnan(out)].tolist()))}")
    return out


def kinematics(T: np.ndarray, pid: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, z_mm: np.ndarray):
    """(E, px, py, pz) float32 of particles with kinetic energy T (float32), see KINEMATICS."""
    m = lookup(PDG_MASS_GEV, np.abs(pid), "mass")
    t = T.astype(np.float64)
    p = np.sqrt(t * (t + 2.0 * m))
    pos = np.stack([x_mm, y_mm, z_mm]).astype(np.float64)
    unit = pos / np.linalg.norm(pos, axis=0)
    px, py, pz = ((p * unit[k]).astype(np.float32) for k in range(3))
    return (t + m).astype(np.float32), px, py, pz


def read_parquet(path: Path, max_row_groups=None) -> dict:
    """Flat numpy arrays of the needed RecHitHGC / SimCluster branches, their per-event counts and event_id."""
    pf = pq.ParquetFile(path)
    n_rg = pf.num_row_groups if max_row_groups is None else min(max_row_groups, pf.num_row_groups)
    parts = {"hits": {k: [] for k in RECHIT_BRANCHES}, "scs": {k: [] for k in SIMCLUSTER_BRANCHES},
             "n_hits": [], "n_sc": [], "event_id": []}
    for rg in range(n_rg):
        table = pf.read_row_group(rg, columns=["RecHitHGC", "SimCluster", "event_id"])
        for collection, branches, out, counts in (("RecHitHGC", RECHIT_BRANCHES, parts["hits"], parts["n_hits"]),
                                                  ("SimCluster", SIMCLUSTER_BRANCHES, parts["scs"], parts["n_sc"])):
            struct = table.column(collection).combine_chunks()
            lengths = None
            for key, branch in branches.items():
                lists = struct.field(branch)
                n = lists.value_lengths().to_numpy(zero_copy_only=False)
                if lengths is None:
                    lengths = n
                elif not np.array_equal(n, lengths):
                    raise ValueError(f"{path.name} row group {rg}: {branch} lengths differ from the other branches")
                out[key].append(lists.flatten().to_numpy(zero_copy_only=False))
            counts.append(lengths.astype(np.int64))
        parts["event_id"].append(table.column("event_id").to_numpy())
    raw = {k: np.concatenate(v) for k, v in parts["hits"].items()}
    raw.update({f"sc_{k}": np.concatenate(v) for k, v in parts["scs"].items()})
    for k in ("n_hits", "n_sc", "event_id"):
        raw[k] = np.concatenate(parts[k])
    return raw


def event_scheme(event_id: np.ndarray, file_index: int) -> str:
    rows = np.arange(len(event_id), dtype=np.int64)
    if np.array_equal(event_id, file_index * EVENTS_PER_FILE + rows):
        return "global (event_id == file_index*1000 + row in file)"
    if np.array_equal(event_id, rows):
        return "per-file (event_id == row in file)"
    raise ValueError(f"event_id fits neither scheme (first {event_id[:5].tolist()}, file_index {file_index})")


def build(raw: dict, file_index: int) -> tuple[dict, dict]:
    """The four structured arrays of one file and bookkeeping counts."""
    n_hits, n_sc = raw["n_hits"], raw["n_sc"]
    n_events, n_rows, n_scs = len(n_hits), int(n_hits.sum()), int(n_sc.sum())
    if n_hits.max() > UINT16_MAX + 1 or n_sc.max() > UINT16_MAX:
        raise ValueError(f"uint16 overflow: max hits/event {n_hits.max()}, max SimClusters/event {n_sc.max()}")
    event = (file_index * EVENTS_PER_FILE + np.arange(n_events)).astype(np.uint64)
    ev_of_hit = np.repeat(np.arange(n_events), n_hits)
    hit_start = np.concatenate([[0], np.cumsum(n_hits)[:-1]])
    hit = (np.arange(n_rows) - hit_start[ev_of_hit]).astype(np.uint16)
    sc_start = np.concatenate([[0], np.cumsum(n_sc)[:-1]])
    ev_of_sc = np.repeat(np.arange(n_events), n_sc)

    x, y, z = mm(raw["x"]), mm(raw["y"]), mm(raw["z"])
    energy = raw["energy"].astype(np.float32)
    idx = raw["idx"].astype(np.int64)
    matched = idx >= 0
    if (idx >= n_sc[ev_of_hit]).any():
        raise ValueError("BestMatchIdx points past the SimCluster list of its event")
    gsc = np.where(matched, sc_start[ev_of_hit] + np.where(matched, idx, 0), 0)  # global SimCluster row per hit
    unc = (np.sqrt(raw["area"].astype(np.float64)) * 10.0 / np.sqrt(12.0)).astype(np.float32)

    ipx, ipy, ipz = mm(raw["sc_ipx"]), mm(raw["sc_ipy"]), mm(raw["sc_ipz"])
    pid = raw["sc_pid"].astype(np.int32)
    E, px, py, pz = kinematics(raw["sc_T"], pid, ipx, ipy, ipz)
    charge = lookup(PDG_CHARGE, pid, "charge").astype(np.int8)

    meas = np.zeros(n_rows, dtype=MEASUREMENTS_DTYPE)
    meas["event"] = event[ev_of_hit]
    meas["hit"] = hit
    meas["x"], meas["y"], meas["z"] = x, y, z
    meas["deposit_energy"] = energy
    meas["x_unc"], meas["y_unc"] = unc, unc

    lab = np.zeros(n_rows, dtype=LABELS_DTYPE)
    for k in KEYS:
        lab[k] = meas[k]
    lab["x"], lab["y"], lab["z"] = x, y, z
    lab["deposit_energy"] = np.where(matched, energy, np.float32(0))
    lab["n_contributions"] = matched.astype(np.uint16)
    lab["con0_particle"] = np.where(matched, idx, 0).astype(np.uint16)
    for name, values in (("x", ipx), ("y", ipy), ("z", ipz), ("px", px), ("py", py), ("pz", pz), ("energy", E)):
        lab[f"con0_{name}"] = np.where(matched, values[gsc], np.float32(0))
    lab["con0_deposit_energy"] = lab["deposit_energy"]
    lab["con0_pid"] = np.where(matched, pid[gsc], 0)

    evl = np.zeros(n_scs, dtype=EVENT_LABEL_DTYPE)
    evl["event"] = event[ev_of_sc]
    evl["particle"] = (np.arange(n_scs) - sc_start[ev_of_sc]).astype(np.uint16)
    evl["px"], evl["py"], evl["pz"], evl["energy"] = px, py, pz, E
    evl["charge"] = charge
    evl["pid"] = pid

    layer = raw["layer"]
    if layer.min() < 0 or layer.max() > np.iinfo(np.uint8).max:
        raise ValueError(f"layer outside uint8: {layer.min()}..{layer.max()}")
    extra = np.zeros(n_rows, dtype=TOY_EXTRA_DTYPE)
    extra["event"], extra["hit"] = meas["event"], hit
    extra["layer"] = layer.astype(np.uint8)
    extra["area"] = raw["area"].astype(np.float32)
    extra["sensor"] = raw["sensor"].astype(np.int32)

    counts = {"events": n_events, "rows": n_rows, "matched_hits": int(matched.sum()),
              "noise_hits": int((~matched).sum()), "simclusters": n_scs,
              "hits_per_event_min": int(n_hits.min()), "hits_per_event_max": int(n_hits.max()),
              "simclusters_per_event_max": int(n_sc.max()), "merged_simclusters": int(raw["sc_merged"].sum())}
    return {"measurements": meas, "labels": lab, "event_label": evl, "toy_extra": extra}, counts


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def validate(npz_path: Path, parquet_path: Path, file_index: int, naming: str,
             max_row_groups=None) -> tuple[list, dict]:
    """Round-trip check of one written npz against a second, independent read of its parquet file (each array's
    stored dtype must be the `naming` dtype; the checks read it under the paper names)."""
    pf = pq.ParquetFile(parquet_path)
    n_rg = pf.num_row_groups if max_row_groups is None else min(max_row_groups, pf.num_row_groups)
    table = pf.read_row_groups(list(range(n_rg)), columns=["RecHitHGC", "SimCluster"])

    def column(collection, branch):
        return pc.list_flatten(pc.struct_field(table.column(collection), branch)).to_numpy()

    def lengths(collection, branch):
        return pc.list_value_length(pc.struct_field(table.column(collection), branch)).to_numpy().astype(np.int64)

    n_hits = lengths("RecHitHGC", "RecHitHGC_x")
    n_sc = lengths("SimCluster", "SimCluster_pdgId")
    n_events, n_rows = len(n_hits), int(n_hits.sum())
    ev_of_hit = np.repeat(np.arange(n_events), n_hits)
    ev_of_sc = np.repeat(np.arange(n_events), n_sc)
    sc_start = np.concatenate([[0], np.cumsum(n_sc)[:-1]])
    event_ids = (file_index * EVENTS_PER_FILE + np.arange(n_events)).astype(np.uint64)
    want = {"x": mm(column("RecHitHGC", "RecHitHGC_x")), "y": mm(column("RecHitHGC", "RecHitHGC_y")),
            "z": mm(column("RecHitHGC", "RecHitHGC_z")), "deposit_energy": column("RecHitHGC", "RecHitHGC_energy")}
    idx = column("RecHitHGC", "RecHitHGC_SimClusterBestMatchIdx").astype(np.int64)
    matched = idx >= 0
    gsc = np.where(matched, sc_start[ev_of_hit] + np.where(matched, idx, 0), 0)
    pid = column("SimCluster", "SimCluster_pdgId").astype(np.int32)
    T = column("SimCluster", "SimCluster_boundaryEnergy")
    tae = column("SimCluster", "SimCluster_truthAssistedEnergy").astype(np.float64)
    fails, stats = [], {}

    def need(ok, message):
        if not ok:
            fails.append(f"{npz_path.name}: {message}")

    with np.load(npz_path, allow_pickle=False) as npz:
        need(sorted(npz.files) == sorted(ARRAYS), f"arrays {sorted(npz.files)}")
        meas = npz["measurements"]
        need(meas.dtype == named_dtype("measurements", naming) and len(meas) == n_rows,
             f"measurements dtype/rows {len(meas)}")
        meas = paper_view(meas, "measurements")
        for k in ("x", "y", "z", "deposit_energy"):
            need(bits_equal(np.ascontiguousarray(meas[k]), want[k]), f"measurements.{k} differs bit-wise from the parquet")
        need(np.array_equal(meas["event"], event_ids[ev_of_hit]), "measurements.event")
        need(np.array_equal(meas["hit"].astype(np.int64),
                            np.arange(n_rows) - np.concatenate([[0], np.cumsum(n_hits)[:-1]])[ev_of_hit]),
             "measurements.hit is not the row index within the event")
        need(not meas["subsystem"].any() and not meas["sample"].any(), "subsystem/sample not all 0")
        need(bool(np.all(np.diff(meas["event"].astype(np.int64)) >= 0)), "rows not sorted by event")
        for k in ("time", "z_unc", "deposit_energy_unc", "time_unc"):
            need(not meas[k].any(), f"measurements.{k} not all 0")
        area = column("RecHitHGC", "RecHitHGC_area")
        unc = (np.sqrt(area.astype(np.float64)) * 10.0 / np.sqrt(12.0)).astype(np.float32)
        need(bits_equal(np.ascontiguousarray(meas["x_unc"]), unc) and bits_equal(np.ascontiguousarray(meas["y_unc"]), unc),
             "x_unc/y_unc")
        keys = {k: meas[k].copy() for k in KEYS}
        del meas

        lab = npz["labels"]
        need(lab.dtype == named_dtype("labels", naming) and len(lab) == n_rows, "labels dtype/rows")
        lab = paper_view(lab, "labels")
        need(all(np.array_equal(lab[k], keys[k]) for k in KEYS), "labels key columns differ from measurements")
        need(int(lab["n_contributions"].astype(np.int64).sum()) == int(matched.sum()),
             "sum of n_contributions != number of matched hits")
        need(np.array_equal(lab["n_contributions"], matched.astype(np.uint16)), "n_contributions != matched flag")
        for k in ("x", "y", "z"):
            need(bits_equal(np.ascontiguousarray(lab[k]), want[k]), f"labels.{k} != sensor position")
        dep = np.where(matched, want["deposit_energy"], np.float32(0))
        need(bits_equal(np.ascontiguousarray(lab["deposit_energy"]), dep), "labels.deposit_energy")
        need(bits_equal(np.ascontiguousarray(lab["con0_deposit_energy"]), dep), "labels.con0_deposit_energy")
        need(np.array_equal(lab["con0_particle"].astype(np.int64), np.where(matched, idx, 0)), "con0_particle")
        need(np.array_equal(lab["con0_pid"], np.where(matched, pid[gsc], 0)), "con0_pid")
        for k, b in (("x", "SimCluster_impactPoint_x"), ("y", "SimCluster_impactPoint_y"),
                     ("z", "SimCluster_impactPoint_z")):
            need(bits_equal(np.ascontiguousarray(lab[f"con0_{k}"]),
                            np.where(matched, mm(column("SimCluster", b))[gsc], np.float32(0))), f"con0_{k}")
        need(all(not lab[f"con{n}_{name}"].any() for n in range(1, N_CONTRIB) for name, _ in CONTRIB_FIELDS)
             and not lab["time"].any() and not lab["con0_time"].any() and not lab["con0_parent_pid"].any()
             and not lab["con0_parent_energy"].any(), "slots 1-2 / time / parent fields not all 0")
        # truth-assisted energy: the builder's own sum of the particle's calibrated hit energies
        summed = np.bincount(gsc[matched], weights=lab["deposit_energy"][matched].astype(np.float64),
                             minlength=len(tae))
        rel = np.abs(summed - tae) / np.where(tae > 0, tae, 1.0)
        stats["truth_assisted_rel_diff_max"] = float(rel.max())
        stats["truth_assisted_rel_diff_mean"] = float(rel.mean())
        con0_energy, con0_pid = lab["con0_energy"].copy(), lab["con0_pid"].copy()
        con0_particle = lab["con0_particle"].astype(np.int64)
        del lab

        evl = npz["event_label"]
        need(evl.dtype == named_dtype("event_label", naming) and len(evl) == int(n_sc.sum()), "event_label dtype/rows")
        evl = paper_view(evl, "event_label")
        need(np.array_equal(evl["event"], event_ids[ev_of_sc]), "event_label.event")
        need(np.array_equal(evl["particle"].astype(np.int64), np.arange(len(evl)) - sc_start[ev_of_sc]),
             "event_label.particle is not the SimCluster row within the event")
        need(np.array_equal(evl["pid"], pid), "event_label.pid")
        # the join labels.con0_particle -> event_label (same event) returns the hit's own particle
        row = sc_start[ev_of_hit] + con0_particle
        need(np.array_equal(evl["pid"][row][matched], con0_pid[matched]), "join con0_particle -> event_label pid")
        need(bits_equal(evl["energy"][row][matched], con0_energy[matched]), "join con0_particle -> event_label energy")
        need(np.array_equal(evl["charge"], lookup(PDG_CHARGE, pid, "charge").astype(np.int8)), "charge")
        mass = lookup(PDG_MASS_GEV, np.abs(pid), "mass")
        t_back = evl["energy"].astype(np.float64) - mass
        t32 = T.astype(np.float64)
        need(bool(np.all(t32 > 0)), "a SimCluster has boundaryEnergy <= 0")
        stats["kinetic_from_energy_nonpositive"] = int(np.sum(t_back <= 0))
        stats["kinetic_from_energy_rel_diff_max"] = float(np.max(np.abs(t_back - t32) / np.where(t32 > 0, t32, 1.0)))
        stats["log10_kinetic_abs_diff_max"] = float(np.max(np.abs(np.log10(np.maximum(t_back, 1e-30))
                                                                  - np.log10(np.maximum(t32, 1e-30)))))
        p2 = sum(evl[k].astype(np.float64) ** 2 for k in ("px", "py", "pz"))
        e2m2 = evl["energy"].astype(np.float64) ** 2 - mass ** 2
        stats["p2_vs_E2_minus_m2_rel_diff_max"] = float(np.max(np.abs(p2 - e2m2) / np.maximum(e2m2, 1e-30)))
        need(not any(evl[k].any() for k in ("Vx", "Vy", "Vz", "time", "parentnumbers", "parentpid", "parentindex")),
             "event_label vertex/time/parent fields not all 0")
        del evl

        extra = npz["toy_extra"]
        need(extra.dtype == named_dtype("toy_extra", naming) and len(extra) == n_rows, "toy_extra dtype/rows")
        extra = paper_view(extra, "toy_extra")
        need(np.array_equal(extra["event"], keys["event"]) and np.array_equal(extra["hit"], keys["hit"]),
             "toy_extra keys")
        need(np.array_equal(extra["layer"].astype(np.int64), column("RecHitHGC", "RecHitHGC_layer").astype(np.int64)),
             "toy_extra.layer")
        need(bits_equal(np.ascontiguousarray(extra["area"]), area), "toy_extra.area")
        need(np.array_equal(extra["sensor"], column("RecHitHGC", "RecHitHGC_detId").astype(np.int32)),
             "toy_extra.sensor")
    stats["n_contributions_sum"] = int(matched.sum())
    return fails, stats


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_metadata(out_dir: Path, set_name: str, summary: dict, naming: str) -> None:
    if naming == "bic":
        text = " ".join(t for docs in FIELD_DOCS.values() for t in docs.values())
        unused = [old for old, _ in BIC_DOC_EDITS if old not in text]
        if unused:
            raise ValueError(f"BIC_DOC_EDITS not found in FIELD_DOCS: {unused}")
    docs = {name: field_docs(name, naming) for name in ARRAYS}
    for name in ARRAYS:
        missing = set(named_dtype(name, naming).names) ^ set(docs[name])
        if missing:
            raise ValueError(f"field docs of {name} do not match the {naming} dtype: {sorted(missing)}")
    schemes = sorted({f["event_id_scheme"] for f in summary["files"]})
    lines = [f"# Toy calorimeter `{set_name}` set in the AI-ready npz format (arXiv:2606.07667)", "",
             f"Written by ssl_toy_interface/B/scripts/toy_to_npz.py on {summary['utc_written']} from "
             f"{summary['input_dir']} ({len(summary['files'])} files). One npz per parquet file "
             f"({set_name}_NNN.parquet -> {set_name}_NNN.npz, 1,000 events each), stored as structured NumPy "
             "arrays with np.savez_compressed: `measurements`, `labels`, `event_label` (the paper's Tables III-V; "
             "the event-level array is named `event_label`, as in the BIC files read by notebooks/0_data_dis.ipynb) "
             "and `toy_extra` (toy-only fields, not part of the paper's schema).", "",
             "The toy (hg_toy_calorimeter, ml4reco/toy_replication) is a silicon sampling calorimeter covering ONE "
             "endcap (front face z = 3200 mm), single-sample, without timing, with one stored truth particle per "
             "hit (the largest depositor of the sensor). Fields the toy cannot fill are 0 and flagged NOT ASSIGNED "
             f"below. Contributor slots: N = 3 as in the BIC files, only {'con0' if naming == 'paper' else 'slot 0'} "
             "is ever filled.", ""]
    if naming == "bic":
        lines += ["Field names and integer types are those of the ePIC BIC files (`amplitude`, `amplitude_unc`; labels "
                  "`Deposit_energy` and `con_index{n}` .. `con_parentEnergy{n}`; event_label `Energy`, with `particle`, "
                  "`charge`, `parentnumbers` and `parentindex` as int32), the naming chosen as canonical on 2026-10-30. "
                  "Each renamed or retyped field states its name or type in the paper's Tables III-V.", ""]
    lines += [f"Event index scheme found in the parquet event_id: {'; '.join(schemes)}.", "",
              "Source files (sha256):", ""]
    lines += [f"- {f['parquet']} {f['parquet_sha256']} -> {f['npz']}" for f in summary["files"]]
    titles = {"measurements": "1. Metadata for the measurements array", "labels": "2. Metadata for the labels array",
              "event_label": "3. Metadata for the event_label array",
              "toy_extra": "4. Metadata for the toy_extra array (toy-only, not in the paper's schema)"}
    for name in ARRAYS:
        dtype = named_dtype(name, naming)
        lines += ["", f"## {titles[name]}", "", f"numpy dtype: `{dtype.descr}`", ""]
        lines += [f"- **{field}:** {docs[name][field]}" for field in dtype.names]
    (out_dir / "metadata.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--set-name", default="pufree")
    ap.add_argument("--files", nargs="*", default=None, help="parquet file names (default: every <set>_NNN.parquet)")
    ap.add_argument("--max-row-groups", type=int, default=None, help="smoke: convert only the first row groups")
    ap.add_argument("--naming", choices=NAMINGS, required=True,
                    help="field names written: bic (the BIC files', canonical) or paper (Tables III-V)")
    args = ap.parse_args()
    if socket.gethostname().startswith("falcon"):
        print("REFUSED: falcon is the login node; run this through sbatch", file=sys.stderr)
        return 3
    names = args.files or sorted(p.name for p in args.in_dir.glob(f"{args.set_name}_[0-9][0-9][0-9].parquet"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"input_dir": str(args.in_dir), "out_dir": str(args.out_dir), "set_name": args.set_name,
               "max_row_groups": args.max_row_groups, "host": socket.gethostname(), "files": [],
               "numpy": np.__version__, "naming": args.naming,
               "dtypes": {k: str(named_dtype(k, args.naming).descr) for k in ARRAYS}}
    all_fails = []
    for name in names:
        src = args.in_dir / name
        k = file_index_of(src)
        dst = args.out_dir / f"{args.set_name}_{k:03d}.npz"
        partial = dst.with_name(dst.name + ".partial")
        t0 = time.time()
        src_sha = sha256(src)
        t1 = time.time()
        raw = read_parquet(src, args.max_row_groups)
        scheme = event_scheme(raw["event_id"], k)
        t2 = time.time()
        arrays, counts = build(raw, k)
        del raw
        arrays = {a: to_naming(v, a, args.naming) for a, v in arrays.items()}
        t3 = time.time()
        with open(partial, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        nbytes = {a: int(v.nbytes) for a, v in arrays.items()}
        del arrays
        t4 = time.time()
        fails, stats = validate(partial, src, k, args.naming, args.max_row_groups)
        t5 = time.time()
        entry = {"parquet": name, "parquet_bytes": src.stat().st_size, "parquet_sha256": src_sha,
                 "npz": dst.name, "npz_bytes": partial.stat().st_size, "uncompressed_bytes": nbytes,
                 "event_id_scheme": scheme, **counts, **stats,
                 "seconds": {"hash": round(t1 - t0, 1), "read": round(t2 - t1, 1), "build": round(t3 - t2, 1),
                             "write_compressed": round(t4 - t3, 1), "validate": round(t5 - t4, 1),
                             "convert_total_excl_validate": round(t4 - t1, 1)},
                 "check": "PASSED" if not fails else "FAILED", "check_failures": fails}
        summary["files"].append(entry)
        if fails:
            all_fails += fails
            print(f"{name}: CHECK FAILED, left as {partial.name}: {fails}", flush=True)
        else:
            os.replace(partial, dst)
            print(f"{name} -> {dst.name}: {counts['events']} events, {counts['rows']:,} rows, "
                  f"{entry['npz_bytes'] / 1e6:.1f} MB, convert {entry['seconds']['convert_total_excl_validate']} s, "
                  f"validate {entry['seconds']['validate']} s, truthAssisted rel diff max "
                  f"{stats['truth_assisted_rel_diff_max']:.2e}", flush=True)
    summary["utc_written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    summary["check"] = "PASSED" if not all_fails else "FAILED"
    write_metadata(args.out_dir, args.set_name, summary, args.naming)
    (args.out_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"CONVERSION {summary['check']}: {len(names)} files, {len(all_fails)} failures, naming {args.naming}",
          flush=True)
    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
