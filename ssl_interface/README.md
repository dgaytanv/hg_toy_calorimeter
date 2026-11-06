# Toy calorimeter -> ssl_hep_np

This directory holds the code that takes events produced by this simulation and feeds them to
`ssl_hep_np` (eicweb.phy.anl.gov/gm-foundation/ssl_hep_np), the self-supervised point-cloud pipeline
written for the AI-ready data format of arXiv:2606.07667. Nothing elsewhere in this repository is changed.

## What the two sides look like

**Our side: the toy parquet.** The event builder of this repository writes events which we convert to
parquet (branch `toy-calorimeter` of github.com/cms-cmu/hgcal-offline-reco). One row per event, two struct
columns:

- `RecHitHGC`: one entry per fired sensor with `energy` (GeV, calibrated), `x`, `y`, `z` (cm), `layer`
  (0..55), `area` (cm^2), `detId`, and `SimClusterBestMatchIdx`, the index of the particle that made the hit
  (-1 for noise).
- `SimCluster`: one entry per truth particle with hits: `pdgId`, `boundaryEnergy` (kinetic energy at the
  calorimeter face), `impactPoint_x/y/z`, `truthAssistedEnergy`, `merged`.

**Their side: the paper's npz schema.** One compressed npz per file with three structured arrays:

- `measurements`: one row per hit, keyed by (`event`, `subsystem`, `hit`, `sample`), with `x`, `y`, `z` (mm),
  the amplitude (`amplitude` in the ePIC BIC files, `deposit_energy` in the paper's Table III), `time`,
  and per-coordinate uncertainties.
- `labels`: row-aligned to `measurements`, the truth per hit: `n_contributions` and up to N contributor slots
  (particle index, position, deposited energy, momentum, PDG id, parent).
- `event_label`: one row per truth particle (index, PDG id, charge, energy, momentum, parents).

The loader in `ssl_hep_np` reads `measurements`, takes four columns [amplitude, x, y, z] per hit, normalises them
(log10 of the amplitude standardised, positions divided by a length scale), makes two augmented copies of each
event, and trains the encoder contrastively. Truth arrays are never used for training; we use them only to probe
what the trained encoder knows.

## Route B: convert to the paper's format, then load it with a generic loader (recommended)

### `toy_to_npz.py`: parquet -> npz

```
python toy_to_npz.py --in-dir <parquet dir> --out-dir <npz dir> --set-name pufree --naming bic
```

One npz per parquet file. The mapping:

| npz array / field | from the toy | notes |
|---|---|---|
| `measurements.event` | file index x 1000 + row | checked against the parquet event id |
| `measurements.subsystem`, `sample` | 0, 0 | one calorimeter, one sample per sensor |
| `measurements.hit` | row order inside the event | |
| `measurements.x/y/z` | `RecHitHGC_x/y/z` | mm = float32(float64(cm) x 10) |
| `measurements.amplitude` | `RecHitHGC_energy` | GeV; `deposit_energy` with `--naming paper` |
| `measurements.x_unc`, `y_unc` | `RecHitHGC_area` | sqrt(area)/sqrt(12), square-cell approximation |
| `measurements.time` | not assigned | the toy has no timing; the parquet sentinel is not copied |
| `labels.n_contributions` | 1 if `SimClusterBestMatchIdx >= 0`, else 0 | the toy records one particle per hit |
| `labels.con0_*` (slot 0) | the matched `SimCluster` | particle index, impact point (mm), PDG id, kinetic energy; slots 1-2 empty |
| `event_label` | one row per `SimCluster` | PDG id, charge from the PDG id, E = T + m with T = `boundaryEnergy` |
| `toy_extra` | `layer`, `area`, `detId` | toy-only, outside the schema, row-aligned to `measurements` |

`--naming bic` writes the field names and integer types of the real ePIC BIC files (`amplitude`,
`Deposit_energy`, `con_index0`, ...), `--naming paper` the names of the paper's Tables III-V. Every file is
written under a `.partial` name, re-read, and checked against an independent second read of the parquet
(E, x, y, z bit-for-bit, key columns, contribution counts, the hit-to-particle join, per-particle energy sums)
before it is renamed; the script exits non-zero on any failure. It also writes `metadata.md`, a field-by-field
description in the style of the paper's Appendix A, and `conversion_summary.json`.

### `npz_dataset.py`: the loader (copy to `src/ssl_hep_np/npz_dataset.py`)

Reads any file in the schema, including the ePIC BIC files unchanged. Field names may be the BIC names or the
paper's; both are read as the same field. The main entry points:

```python
from ssl_hep_np.npz_dataset import load_npz_events, NpzMeasurementsDataset, hit_truth, event_truth

events = load_npz_events(files, extra_fields=("x_unc", "y_unc"))      # columns [E, x, y, z, (extras)]
ds = NpzMeasurementsDataset(events, transform=default_augmentation(), normalize=True,
                            norm={"log_e_mean": m, "log_e_std": s, "pos_scale": 1000.0})
```

`ds[i]` is the same `(Data, Data)` view pair that the repository's ColliderML dataset produces, so
`contrastive_collate`, the augmentations and `PointCloudEncoder` are reused unchanged. `samples="sum"` merges
multi-sample hits as the repository's DIS notebook does for the ScFi layer. `hit_truth` and `event_truth` return
the truth arrays for evaluation only.

`test_npz_dataset.py` (copy to `tests/`) covers the alias handling, the sample merging, the normalisation and
the view pairs.

### `ssl_hep_np_dataset_route_B.patch`

The one change `npz_dataset.py` needs in the repository's `dataset.py`: `normalize_hits()` takes the
normalisation constants as arguments instead of the ColliderML constants hard-coded in the file, so a different
detector passes its own. Apply with `git apply ssl_hep_np_dataset_route_B.patch` in the ssl_hep_np clone.

## Route A: read the parquet directly

`toy_dataset.py` (copy to `src/ssl_hep_np/toy_dataset.py`) skips the conversion: it reads the toy parquet into
per-event arrays and serves the same view pairs. It also offers extra per-hit inputs (`layer`, `area`) through
`feature_set="C1"`. `ssl_hep_np_dataset_route_A.patch` is the matching `dataset.py` change (the same
parameterised `normalize_hits`, plus `compute_norm_constants` to derive the energy normalisation from the
training split).

Both routes were run on the same 5,000 events and give bit-identical [E, x, y, z] tensors and the same
training curves and probe scores. Route B is the one to keep: it makes the toy a second detector in the
collaboration's format, and the loader is the same code that reads their files.

## Normalisation constants

The repository's constants are for ColliderML. For the toy we compute them on the training split:
mean and standard deviation of log10(E [GeV]) over all hits (about -1.50 and 1.00), with the position scale
kept at 1000 mm. Both routes expose a function for this (`compute_norm_constants`).
