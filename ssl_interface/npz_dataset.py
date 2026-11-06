"""Datasets in the AI-ready npz format of arXiv:2606.07667 (arrays measurements / labels / event_label).

One loader for any detector written in that format, e.g. the ePIC BIC files read in notebooks/0_data_dis.ipynb or
the toy calorimeter converted to it. The canonical field names are those of the BIC files (amplitude, Deposit_energy,
con_pid0, Energy, ...), chosen on 2026-10-30; the names of the paper's Tables III-V (deposit_energy, con0_pid,
energy, ...) are read as aliases of them, so files written with either naming load the same way.
"""

import re

import numpy as np
import torch
import torch.utils.data as data
from torch_geometric.data import Data

from .dataset import normalize_hits


# ── Format ─────────────────────────────────────────────────────────────────

KEY_FIELDS = ("event", "subsystem", "hit", "sample")
# Per array, canonical (BIC file) name -> the names read as that field, canonical first. Per array because the
# paper's deposit_energy is `amplitude` in measurements but `Deposit_energy` in labels.
FIELD_ALIASES = {
    "measurements": {"amplitude": ("amplitude", "deposit_energy"),
                     "amplitude_unc": ("amplitude_unc", "deposit_energy_unc")},
    "labels": {"Deposit_energy": ("Deposit_energy", "deposit_energy")},
    "event_label": {"Energy": ("Energy", "energy")},
}
SLOT_ALIASES = {   # labels contributor slot n: BIC con_<key><n>, paper con<n>_<value>
    "index": "particle", "x": "x", "y": "y", "z": "z", "DepositEnergy": "deposit_energy", "time": "time",
    "px": "px", "py": "py", "pz": "pz", "Energy": "energy", "pid": "pid", "parentpid": "parent_pid",
    "parentEnergy": "parent_energy",
}
_SLOT_BIC = {paper: bic for bic, paper in SLOT_ALIASES.items()}
PLACEHOLDER_SUBSYSTEM = 65535   # the BIC files mark an event without hits by one row with this subsystem


def field_names(array, name):
    """Names read as field `name` of `array` ("measurements", "labels" or "event_label"), canonical first."""
    for names in FIELD_ALIASES.get(array, {}).values():
        if name in names:
            return names
    if array == "labels":
        m = re.fullmatch(r"con_([A-Za-z]+)(\d+)", name)
        if m and m.group(1) in SLOT_ALIASES:
            return name, f"con{m.group(2)}_{SLOT_ALIASES[m.group(1)]}"
        m = re.fullmatch(r"con(\d+)_([a-z_]+)", name)
        if m and m.group(2) in _SLOT_BIC:
            return f"con_{_SLOT_BIC[m.group(2)]}{m.group(1)}", name
    return (name,)


def get_field(arr, name, array):
    """Column `name` (canonical or alias) of a structured array of `array`."""
    for candidate in field_names(array, name):
        if candidate in arr.dtype.names:
            return arr[candidate]
    raise KeyError(f"{array} field {name!r} not in {arr.dtype.names}")


def canonical_names(arr, array):
    """View of a structured array of `array` with every field under its canonical name (types as stored)."""
    names = [field_names(array, n)[0] for n in arr.dtype.names]
    if names == list(arr.dtype.names):
        return arr
    fields = arr.dtype.fields
    return arr.view(np.dtype({"names": names, "formats": [fields[n][0] for n in arr.dtype.names],
                              "offsets": [fields[n][1] for n in arr.dtype.names], "itemsize": arr.dtype.itemsize}))


# ── Loading ────────────────────────────────────────────────────────────────


class NpzEvents:
    """Points of many events in one (N, F) float32 array; event i is rows offsets[i]:offsets[i + 1].

    hit_labels: dict of labels columns, row-aligned to features (or None).
    event_label: event_label rows of the kept events, event i is rows label_offsets[i]:label_offsets[i + 1].
    """

    def __init__(self, features, offsets, event_ids, columns, hit_labels=None, event_label=None,
                 label_offsets=None):
        self.features = features
        self.offsets = offsets
        self.event_ids = event_ids
        self.columns = columns
        self.hit_labels = hit_labels
        self.event_label = event_label
        self.label_offsets = label_offsets

    def __len__(self):
        return len(self.event_ids)

    def hits(self, i):
        return self.features[self.offsets[i]:self.offsets[i + 1]]


def _group_samples(meas, order, ev):
    """One row per (event, subsystem, hit), groups in order of their first sample.

    Returns (row of each group's first sample within `order`, group id of every row of `order`).
    """
    key = np.stack([ev.astype(np.uint64), meas["subsystem"][order].astype(np.uint64),
                    meas["hit"][order].astype(np.uint64)], axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    rank = np.argsort(first, kind="stable")
    group_of = np.empty_like(rank)
    group_of[rank] = np.arange(len(rank))
    return first[rank], group_of[inverse.ravel()]


def load_npz_events(files, feature_fields=("amplitude", "x", "y", "z"), extra_fields=(), samples="rows",
                    subsystems=None, max_events=None, label_fields=(), with_event_label=False):
    """Read npz files of the format into one NpzEvents.

    Field names may be canonical or aliases (see FIELD_ALIASES); the NpzEvents reports them under the canonical
    names (columns, hit_labels keys, event_label fields).

    Args:
        files: npz paths (or file objects), read in this order.
        feature_fields: measurements fields of columns 0-3 (E, x, y, z: what the augmentations and
            normalize_hits() expect).
        extra_fields: further measurements fields appended as columns 4, 5, ... (e.g. "x_unc", "y_unc").
        samples: "rows" -> every measurement row is a point (a multi-sample hit gives one point per sample);
            "sum" -> one point per (event, subsystem, hit) with the amplitude summed over its samples and every
            other field taken from its first sample, as notebooks/0_data_dis.ipynb does for ScFi.
        subsystems: keep only these subsystem ids (default: all but the placeholder 65535).
        max_events: keep only the first max_events events over all files.
        label_fields: labels fields returned per point (row-aligned; with samples="sum" the first sample's).
        with_event_label: also return the event_label rows of the kept events.
    """
    if samples not in ("rows", "sum"):
        raise ValueError(f"samples must be 'rows' or 'sum', got {samples!r}")
    columns = tuple(field_names("measurements", c)[0] for c in tuple(feature_fields) + tuple(extra_fields))
    label_keys = tuple(field_names("labels", k)[0] for k in label_fields)
    if len(set(label_keys)) != len(label_keys):
        raise ValueError(f"label_fields name a field twice: {label_fields}")
    feats, offsets, event_ids, labels, evls, label_offsets = [], [0], [], {k: [] for k in label_keys}, [], [0]
    for f in files:
        remaining = None if max_events is None else max_events - sum(len(e) for e in event_ids)
        if remaining is not None and remaining <= 0:
            break
        with np.load(f, allow_pickle=False) as npz:
            meas = npz["measurements"]
            keep = meas["subsystem"] != PLACEHOLDER_SUBSYSTEM
            if subsystems is not None:
                keep &= np.isin(meas["subsystem"], subsystems)
            order = np.flatnonzero(keep)
            ev = meas["event"][order]
            if np.any(ev[1:] < ev[:-1]):
                order = order[np.argsort(ev, kind="stable")]
                ev = meas["event"][order]
            if samples == "sum":
                first, group = _group_samples(meas, order, ev)
                energy = np.bincount(group, weights=get_field(meas, "amplitude", "measurements")[order].astype(
                    np.float64), minlength=len(first)).astype(np.float32)
                order, ev = order[first], ev[first]
            uev, start = np.unique(ev, return_index=True)
            if remaining is not None and len(uev) > remaining:
                stop = start[remaining]
                uev, start, order, ev = uev[:remaining], start[:remaining], order[:stop], ev[:stop]
                if samples == "sum":
                    energy = energy[:stop]
            x = np.empty((len(order), len(columns)), dtype=np.float32)
            for j, c in enumerate(columns):
                if samples == "sum" and c == "amplitude":
                    x[:, j] = energy
                else:
                    x[:, j] = get_field(meas, c, "measurements")[order]
            feats.append(x)
            offsets.extend((offsets[-1] + np.append(start[1:], len(order))).tolist())
            event_ids.append(uev)
            if label_keys:
                lab = npz["labels"]
                if len(lab) != len(meas) or any(not np.array_equal(lab[k], meas[k]) for k in KEY_FIELDS):
                    raise ValueError(f"{f}: labels are not row-aligned with measurements on {KEY_FIELDS}")
                for k in label_keys:
                    labels[k].append(get_field(lab, k, "labels")[order])
                del lab
            if with_event_label:
                evl = canonical_names(npz["event_label"], "event_label")
                evl = evl[np.isin(evl["event"], uev)]
                evl = evl[np.argsort(evl["event"], kind="stable")]
                ends = np.searchsorted(evl["event"], uev, side="right")
                label_offsets.extend((label_offsets[-1] + ends).tolist())
                evls.append(evl)
            del meas
    return NpzEvents(
        features=np.concatenate(feats) if feats else np.empty((0, len(columns)), np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        event_ids=np.concatenate(event_ids) if event_ids else np.empty(0, np.uint64),
        columns=columns,
        hit_labels={k: np.concatenate(v) for k, v in labels.items()} if label_keys else None,
        event_label=np.concatenate(evls) if with_event_label else None,
        label_offsets=np.asarray(label_offsets, dtype=np.int64) if with_event_label else None,
    )


# ── Probe targets (never model inputs) ─────────────────────────────────────

PDG_MASS_GEV = {11: 0.00051099895, 13: 0.1056583755, 211: 0.13957039, 22: 0.0}   # PDG 2024, by |pid|
HIT_CLASSES = ("EM", "hadron", "muon", "noise")
TRUTH_LABEL_FIELDS = ("n_contributions", "con_pid0", "con_Energy0")


def kinetic_energy(energy, pid):
    """energy - m(|pid|) (GeV, float64); NaN where the pid has no entry in PDG_MASS_GEV."""
    mass = np.full(len(pid), np.nan)
    for apid, m in PDG_MASS_GEV.items():
        mass[np.abs(pid) == apid] = m
    return np.asarray(energy, dtype=np.float64) - mass


def hit_truth(events):
    """Per-point probe targets of NpzEvents loaded with label_fields=TRUTH_LABEL_FIELDS.

    hit_class: 0 EM (|pid| 11, 22), 1 hadron (|pid| 211), 2 muon (|pid| 13), 3 noise (n_contributions 0), -1 other.
    log10_kinetic: log10 of the kinetic energy of the hit's leading particle, con_Energy0 - m(|con_pid0|) (the toy
        stores total energy at production); NaN for noise.
    """
    lab = events.hit_labels
    apid = np.abs(lab["con_pid0"])
    noise = lab["n_contributions"] == 0
    cls = np.full(len(apid), -1, dtype=np.int8)
    cls[np.isin(apid, (11, 22))] = 0
    cls[apid == 211] = 1
    cls[apid == 13] = 2
    cls[noise] = 3
    kin = kinetic_energy(lab["con_Energy0"], lab["con_pid0"])
    log10_kinetic = np.where(noise, np.nan, np.log10(np.where(noise, 1.0, kin)))
    return {"hit_class": cls, "log10_kinetic": log10_kinetic}


def event_truth(events):
    """Per-event probe targets from event_label: number of particles and the sum of their kinetic energies."""
    evl, off = events.event_label, events.label_offsets
    n = np.diff(off)
    kin = kinetic_energy(evl["Energy"], evl["pid"])
    sums = np.bincount(np.repeat(np.arange(len(n)), n), weights=kin, minlength=len(n))
    return {"n_particles": n, "sum_kinetic": sums}


# ── Dataset ────────────────────────────────────────────────────────────────


class NpzMeasurementsDataset(data.Dataset):
    """Contrastive dataset over NpzEvents: two augmented views per event, as ColliderMLDataset.

    The transform acts on columns 0-3 ([E, x, y, z]); extra columns ride along with their rows. With
    normalize=True, normalize_hits() maps columns 0-3 with `norm` ({log_e_mean, log_e_std, pos_scale}) and extra
    column 4 + j is divided by extra_scale[j].
    """

    def __init__(self, events, transform=None, normalize=False, norm=None, extra_scale=None, labels=None):
        n_extra = events.features.shape[1] - 4
        if n_extra < 0:
            raise ValueError("NpzMeasurementsDataset needs at least the four columns [E, x, y, z]")
        if normalize and n_extra and (extra_scale is None or len(extra_scale) != n_extra):
            raise ValueError(f"normalize=True needs extra_scale for the {n_extra} extra column(s)")
        self.events = events
        self.transform = transform
        self.normalize = normalize
        self.norm = norm or {}
        self.extra_scale = None if extra_scale is None else torch.as_tensor(extra_scale, dtype=torch.float32)
        self.labels = labels
        self._hit_truth = None
        self._event_truth = None

    def __len__(self):
        return len(self.events)

    def normalize_view(self, pc):
        out = normalize_hits(pc, **self.norm)
        if self.extra_scale is not None:
            out[:, 4:] = pc[:, 4:] / self.extra_scale
        return out

    def __getitem__(self, idx):
        pc = torch.from_numpy(self.events.hits(idx))
        view1 = self.transform(pc.clone()) if self.transform else pc.clone()
        view2 = self.transform(pc.clone()) if self.transform else pc.clone()
        if self.normalize:
            view1, view2 = self.normalize_view(view1), self.normalize_view(view2)
        label = self.labels[idx] if self.labels is not None else -1
        return Data(x=view1, y=label), Data(x=view2, y=label)

    def truth(self, idx):
        """Probe targets of event idx (see hit_truth and event_truth); needs labels and event_label loaded."""
        if self._hit_truth is None:
            self._hit_truth, self._event_truth = hit_truth(self.events), event_truth(self.events)
        sl = slice(self.events.offsets[idx], self.events.offsets[idx + 1])
        out = {k: v[sl] for k, v in self._hit_truth.items()}
        out.update({k: v[idx] for k, v in self._event_truth.items()})
        return out
