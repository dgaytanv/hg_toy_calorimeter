# Toy calorimeter -> ssl_hep_np (AI-ready QCD data format of arXiv:2606.07667)

Code that feeds the toy-calorimeter events (the parquet files produced from this simulation) into the
`ssl_hep_np` contrastive pipeline (eicweb.phy.anl.gov/gm-foundation/ssl_hep_np). Two routes were built and compared;
both give bit-identical [E, x, y, z] tensors.

Route B, the paper's format (recommended):

| file | role |
|---|---|
| `toy_to_npz.py` | toy parquet (RecHitHGC / SimCluster struct columns, one row per event) -> one npz per file with the paper's `measurements` / `labels` / `event_label` arrays (`--naming bic` writes the ePIC BIC field names, `--naming paper` the names of Tables III-V); self-checking, writes `metadata.md` and `conversion_summary.json` |
| `npz_dataset.py` | drop into `src/ssl_hep_np/`: a loader for any detector in that format (reads the ePIC BIC files unchanged, and the converted toy files), serving the same view pairs as the repo's ColliderML loader |
| `test_npz_dataset.py` | its unit tests (`tests/`) |
| `ssl_hep_np_dataset_route_B.patch` | the small change to `src/ssl_hep_np/dataset.py` it needs (normalisation constants as parameters instead of ColliderML constants) |

Route A, direct parquet adapter:

| file | role |
|---|---|
| `toy_dataset.py` | drop into `src/ssl_hep_np/`: reads the toy parquet straight into the pipeline, with optional extra per-hit inputs (layer, cell area) |
| `ssl_hep_np_dataset_route_A.patch` | the matching change to `src/ssl_hep_np/dataset.py` (parameterised `normalize_hits`, `compute_norm_constants`) |

Units: toy positions are cm and are converted as mm = float32(float64(cm) * 10.0). Truth columns are carried for
evaluation only and never enter the encoder.
