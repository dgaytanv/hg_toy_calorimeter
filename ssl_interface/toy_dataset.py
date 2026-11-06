"""Toy-calorimeter (HGCAL-like endcap) parquet adapter for the contrastive pipeline.

Reads the toy_replication production parquet (struct columns RecHitHGC and SimCluster, one row per event) straight
into per-event numpy arrays and serves the same (Data, Data) view pairs as ColliderMLDataset, so contrastive_collate,
the augmentations and PointCloudEncoder are reused unchanged.
"""

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.utils.data as data
from torch_geometric.data import Data

from ssl_hep_np.dataset import Compose, RandomHitDropout, log10_mean_std, normalize_hits


# ── Reading ────────────────────────────────────────────────────────────────

CM_TO_MM = 10.0
LAYER_MAX = 55.0         # toy layers are 0..55
AREA_FLOOR = 1e-6        # cm^2; only guards log10, cell areas are O(1 cm^2)
HIT_FIELDS = ("energy", "x", "y", "z", "layer", "area")
TRUTH_HIT_FIELDS = ("SimClusterBestMatchIdx",)
TRUTH_SC_FIELDS = ("pdgId", "boundaryEnergy")

### per-hit truth classes from the best-matched SimCluster's pdgId (probe targets, never model inputs)
CLASS_NAMES = ("EM", "hadron", "muon", "noise")
EM, HADRON, MUON, NOISE, OTHER = 0, 1, 2, 3, -1
TRUTH_KEYS = ("hit_class", "hit_matched", "hit_match_energy", "sum_boundary_energy", "n_simclusters")
EVENT_TRUTH_KEYS = ("sum_boundary_energy", "n_simclusters")


def cm_to_mm(a):
    """float32(float64(a_cm) * 10.0): the exact cm -> mm conversion of EVAL_PROTOCOL section 2."""
    return (np.asarray(a).astype(np.float64) * CM_TO_MM).astype(np.float32)


def _list_column(struct, name):
    """(values, offsets) of a list-valued struct field; offsets start at 0, event i = values[offsets[i]:offsets[i+1]]."""
    lst = struct.field(name)
    assert lst.null_count == 0 and lst.values.null_count == 0, f"{name}: unexpected nulls"
    offsets = lst.offsets.to_numpy().astype(np.int64)
    values = lst.values.to_numpy(zero_copy_only=False)[offsets[0]:offsets[-1]]
    return values, offsets - offsets[0]


def _truth(table, match_idx, offsets):
    """Per-hit class / matched flag / matched boundaryEnergy and per-event SimCluster totals of one row group."""
    sc = table.column("SimCluster").combine_chunks()
    pdg, sc_offsets = _list_column(sc, "SimCluster_pdgId")
    energy, e_offsets = _list_column(sc, "SimCluster_boundaryEnergy")
    assert np.array_equal(sc_offsets, e_offsets), "SimCluster fields have different lengths"
    n_sc = np.diff(sc_offsets)
    ev_of_hit = np.repeat(np.arange(len(n_sc)), np.diff(offsets))
    idx = match_idx.astype(np.int64)
    matched = idx >= 0
    assert (idx[matched] < n_sc[ev_of_hit[matched]]).all(), "SimClusterBestMatchIdx beyond the event's SimClusters"
    row = np.where(matched, sc_offsets[:-1][ev_of_hit] + idx, 0)   ### flat SimCluster row (0 = placeholder for noise)
    abs_pdg = np.abs(pdg[row]) if pdg.size else np.zeros(idx.size, dtype=np.int64)
    hit_class = np.full(idx.size, OTHER, dtype=np.int8)
    hit_class[np.isin(abs_pdg, (11, 22))] = EM
    hit_class[abs_pdg == 211] = HADRON
    hit_class[abs_pdg == 13] = MUON
    hit_class[~matched] = NOISE
    ev_of_sc = np.repeat(np.arange(len(n_sc)), n_sc)
    return {
        "hit_class": hit_class,
        "hit_matched": matched,
        "hit_match_energy": np.where(matched, energy[row] if energy.size else np.nan, np.nan).astype(np.float32),
        "sum_boundary_energy": np.bincount(ev_of_sc, weights=energy.astype(np.float64), minlength=len(n_sc)),
        "n_simclusters": n_sc,
    }


def _row_group_events(pf, rg, path, with_truth):
    """Per-event dicts of one parquet row group (hit arrays are views into the row group's flat arrays)."""
    hit_fields = HIT_FIELDS + (TRUTH_HIT_FIELDS if with_truth else ())
    columns = ["event_id"] + [f"RecHitHGC.RecHitHGC_{f}" for f in hit_fields]
    if with_truth:
        columns += [f"SimCluster.SimCluster_{f}" for f in TRUTH_SC_FIELDS]
    table = pf.read_row_group(rg, columns=columns)
    rechits = table.column("RecHitHGC").combine_chunks()
    assert rechits.offset == 0
    cols, offsets = {}, None
    for f in hit_fields:
        cols[f], off = _list_column(rechits, f"RecHitHGC_{f}")
        offsets = off if offsets is None else offsets
        assert np.array_equal(off, offsets), f"RecHitHGC_{f}: hit counts differ from RecHitHGC_{hit_fields[0]}"
    arrays = {
        "E": cols["energy"].astype(np.float32, copy=False),
        "x": cm_to_mm(cols["x"]),
        "y": cm_to_mm(cols["y"]),
        "z": cm_to_mm(cols["z"]),
        "layer": cols["layer"],
        "area": cols["area"].astype(np.float32, copy=False),
    }
    if with_truth:
        arrays.update(_truth(table, cols["SimClusterBestMatchIdx"], offsets))
    event_id = table.column("event_id").to_numpy()
    events = []
    for i in range(len(event_id)):
        s, e = offsets[i], offsets[i + 1]
        ev = {"file": str(path), "row_group": rg, "event_id": int(event_id[i])}
        for k, v in arrays.items():
            ev[k] = v[i].item() if k in EVENT_TRUTH_KEYS else v[s:e]
        events.append(ev)
    return events


def load_toy_events(files, n_events=None, with_truth=False):
    """Read toy parquet files into a list of per-event dicts of numpy arrays, in file and row order.

    Reads row group by row group and only the struct fields it needs (no Python lists). Per event: event_id, E (GeV),
    x, y, z (mm, via cm_to_mm), layer, area (cm^2), hits in parquet row order; with_truth adds TRUTH_KEYS.
    n_events: stop after the first n events of `files` (smoke runs).
    """
    events = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            if n_events is not None and len(events) >= n_events:
                return events[:n_events]
            events.extend(_row_group_events(pf, rg, path, with_truth))
    return events if n_events is None else events[:n_events]


def compute_area_constants(events):
    """Mean / std of log10(area [cm^2]) over all hits of `events`: the standardization of C1's area column."""
    n, mean, std = log10_mean_std((ev["area"] for ev in events), floor=AREA_FLOOR)
    return {"LOG_AREA_MEAN": mean, "LOG_AREA_STD": std, "N_HITS": n}


# ── Augmentation of the first columns only ─────────────────────────────────


def augment_columns(transform, pc, n_aug=4):
    """transform applied to columns 0..n_aug-1 of pc; extra columns follow the kept hits untouched.

    With no extra columns this is transform(pc). Otherwise the stages of a Compose run on the first n_aug columns
    one by one, and a RandomHitDropout stage runs on those columns plus a row-index column, so it keeps the same hits
    (same RNG draws) as transform(pc[:, :n_aug]) would and the index selects the extra columns' rows.
    """
    if pc.size(1) == n_aug:
        return transform(pc)
    assert pc.size(0) < 2 ** 24, "row index must be exact in float32"
    stages = transform.transforms if isinstance(transform, Compose) else [transform]
    head, tail = pc[:, :n_aug].clone(), pc[:, n_aug:]
    for stage in stages:
        if isinstance(stage, RandomHitDropout):
            rows = torch.arange(head.size(0), dtype=head.dtype).unsqueeze(1)
            out = stage(torch.cat([head, rows], dim=1))
            head, tail = out[:, :n_aug], tail[out[:, n_aug].long()]
        else:
            n = head.size(0)
            head = stage(head)
            if head.size(0) != n:
                raise ValueError(f"{type(stage).__name__} changed the hit count; only RandomHitDropout may drop hits")
    return torch.cat([head, tail], dim=1)


# ── Dataset ────────────────────────────────────────────────────────────────

FEATURE_SETS = {
    "C0": ("E", "x", "y", "z"),
    "C1": ("E", "x", "y", "z", "layer/55", "standardized log10(area)"),
}


class ToyCaloDataset(data.Dataset):
    """Contrastive dataset over toy events: returns two augmented views per event, like ColliderMLDataset.

    Columns: C0 = [E, x, y, z] (GeV, mm); C1 appends layer / 55 and log10(area [cm^2]) standardized with
    norm["LOG_AREA_MEAN"] / norm["LOG_AREA_STD"]. The transform acts on columns 0-3 only (extra columns follow the
    kept hits untouched); if normalize=True, normalize_hits() with the constants in `norm` (LOG_E_MEAN, LOG_E_STD,
    POS_SCALE) maps columns 0-3 after the augmentations. Probe truth is reachable via truth(idx), never via the views.
    """

    def __init__(self, events, transform=None, normalize=False, feature_set="C0", norm=None, labels=None):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}, got {feature_set!r}")
        if (normalize or feature_set != "C0") and norm is None:
            raise ValueError("norm constants are required for normalize=True and for feature sets beyond C0")
        self.events = events
        self.transform = transform
        self.normalize = normalize
        self.feature_set = feature_set
        self.norm = norm
        self.labels = labels

    @property
    def input_dim(self):
        return len(FEATURE_SETS[self.feature_set])

    def __len__(self):
        return len(self.events)

    def hits(self, idx):
        """(N, F) float32 input of event idx before augmentation and before normalizing columns 0-3."""
        ev = self.events[idx]
        cols = [ev["E"], ev["x"], ev["y"], ev["z"]]
        if self.feature_set == "C1":
            log_area = np.log10(np.maximum(ev["area"].astype(np.float64), AREA_FLOOR))
            cols += [ev["layer"].astype(np.float32) / np.float32(LAYER_MAX),
                     ((log_area - self.norm["LOG_AREA_MEAN"]) / self.norm["LOG_AREA_STD"]).astype(np.float32)]
        return torch.from_numpy(np.stack(cols, axis=1).astype(np.float32, copy=False))

    def normalize_view(self, pc):
        """normalize_hits() on columns 0-3 with this dataset's constants; extra columns are already O(1)."""
        head = normalize_hits(pc[:, :4], log_e_mean=self.norm["LOG_E_MEAN"], log_e_std=self.norm["LOG_E_STD"],
                              pos_scale=self.norm["POS_SCALE"])
        return head if pc.size(1) == 4 else torch.cat([head, pc[:, 4:]], dim=1)

    def model_input(self, idx):
        """Un-augmented, normalized (N, F) input of event idx (frozen-encoder evaluation)."""
        return self.normalize_view(self.hits(idx))

    def __getitem__(self, idx):
        pc = self.hits(idx)
        view1 = augment_columns(self.transform, pc.clone()) if self.transform else pc.clone()
        view2 = augment_columns(self.transform, pc.clone()) if self.transform else pc.clone()
        if self.normalize:
            view1, view2 = self.normalize_view(view1), self.normalize_view(view2)
        label = self.labels[idx] if self.labels is not None else -1
        return Data(x=view1, y=label), Data(x=view2, y=label)

    def truth(self, idx):
        """Probe truth of event idx: per hit hit_class / hit_matched / hit_match_energy (GeV, NaN for noise), per
        event sum_boundary_energy (GeV) / n_simclusters. Needs events loaded with_truth=True."""
        ev = self.events[idx]
        return {k: ev[k] for k in TRUTH_KEYS}
