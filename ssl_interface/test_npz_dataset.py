"""Unit tests of ssl_hep_np.npz_dataset on synthetic BIC-like files (no data on disk).

Run with `python tests/test_npz_dataset.py` (or pytest). The fixture is written in the canonical naming of the BIC
files (amplitude, amplitude_unc, Deposit_energy, con_index0 / con_pid0 / con_Energy0, Energy) or in the paper's
(deposit_energy, deposit_energy_unc, con0_particle / con0_pid / con0_energy, energy), with a two-sample subsystem 1
next to a single-sample subsystem 0, a placeholder event (subsystem 65535) and rows not sorted by event.
"""

import io

import numpy as np
import torch

from ssl_hep_np.dataset import GaussianNoise, default_augmentation, normalize_hits
from ssl_hep_np.npz_dataset import (TRUTH_LABEL_FIELDS, NpzMeasurementsDataset, event_truth, hit_truth,
                                    load_npz_events)

NAMES = {   # canonical name -> the fixture's field name, per naming
    "bic": {"amplitude": "amplitude", "amplitude_unc": "amplitude_unc", "Deposit_energy": "Deposit_energy",
            "con_index0": "con_index0", "con_pid0": "con_pid0", "con_Energy0": "con_Energy0", "Energy": "Energy"},
    "paper": {"amplitude": "deposit_energy", "amplitude_unc": "deposit_energy_unc", "Deposit_energy": "deposit_energy",
              "con_index0": "con0_particle", "con_pid0": "con0_pid", "con_Energy0": "con0_energy", "Energy": "energy"},
}
PARTICLE_TYPE = {"bic": "<i4", "paper": "<u2"}   # event_label particle: int32 in the BIC files

# (event, subsystem, hit, sample, amplitude, n_contributions, con_index0, con_pid0, con_Energy0); x = 10 * row number
ROWS = [
    (7, 0, 0, 0, 1.0, 1, 0, 22, 5.0),
    (7, 1, 0, 0, 2.0, 1, 1, 211, 3.0),
    (7, 1, 0, 1, 3.0, 1, 1, 211, 3.0),
    (7, 0, 1, 0, 4.0, 0, 0, 0, 0.0),
    (7, 1, 1, 0, 5.0, 1, 2, 13, 1.0),
    (7, 1, 1, 1, 6.0, 1, 2, 13, 1.0),
    (3, 65535, 0, 0, 0.0, 0, 0, 0, 0.0),   # placeholder: event 3 has no hits
    (5, 1, 0, 0, 7.0, 1, 0, 11, 2.0),
    (5, 0, 0, 0, 8.0, 1, 1, 22, 4.0),
    (5, 1, 0, 1, 9.0, 1, 0, 11, 2.0),
]


def dtypes(naming):
    n = NAMES[naming]
    meas = np.dtype([("event", "<u8"), ("subsystem", "<u2"), ("hit", "<u2"), ("sample", "<u2"),
                     ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), (n["amplitude"], "<f4"), ("time", "<f4"),
                     ("x_unc", "<f4"), ("y_unc", "<f4"), ("z_unc", "<f4"), (n["amplitude_unc"], "<f4"),
                     ("time_unc", "<f4")])
    lab = np.dtype([("event", "<u8"), ("subsystem", "<u2"), ("hit", "<u2"), ("sample", "<u2"),
                    (n["Deposit_energy"], "<f4"), ("n_contributions", "<u2"), (n["con_index0"], "<u2"),
                    (n["con_pid0"], "<i4"), (n["con_Energy0"], "<f4")])
    evl = np.dtype([("event", "<u8"), ("particle", PARTICLE_TYPE[naming]), (n["Energy"], "<f4"), ("pid", "<i4")])
    return meas, lab, evl


def fixture_bytes(misalign_labels=False, naming="bic"):
    meas_type, lab_type, evl_type = dtypes(naming)
    meas = np.zeros(len(ROWS), meas_type)
    lab = np.zeros(len(ROWS), lab_type)
    for i, (ev, sub, hit, smp, amp, ncon, idx, pid, en) in enumerate(ROWS):
        meas[i] = (ev, sub, hit, smp, 10.0 * i, 0.5, 100.0, amp, 25.0 * smp, i + 1.0, 2.0 * (i + 1.0), 0.0,
                   0.1 * amp, 0.0)
        lab[i] = (ev, sub, hit, smp, amp if ncon else 0.0, ncon, idx, pid, en)
    if misalign_labels:
        lab["hit"][0] = 9
    evl = np.array([(7, 0, 5.0, 22), (7, 1, 3.0, 211), (7, 2, 1.0, 13), (5, 0, 2.0, 11), (5, 1, 4.0, 22)],
                   evl_type)
    buf = io.BytesIO()
    np.savez_compressed(buf, measurements=meas, labels=lab, event_label=evl)
    return buf.getvalue()


def load(**kw):
    raw = fixture_bytes(kw.pop("misalign", False), kw.pop("naming", "bic"))
    return load_npz_events([io.BytesIO(raw)], **kw)


def test_rows_sort_and_placeholder():
    ev = load()
    assert ev.event_ids.tolist() == [5, 7]                      # placeholder event 3 dropped, events sorted
    assert ev.offsets.tolist() == [0, 3, 9]
    # stable sort keeps the in-event row order
    assert ev.features[:, 0].tolist() == [7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert ev.features[:, 1].tolist() == [70.0, 80.0, 90.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert ev.columns == ("amplitude", "x", "y", "z")


def test_sum_over_samples():
    ev = load(samples="sum", extra_fields=("amplitude_unc",))
    assert ev.offsets.tolist() == [0, 2, 6]
    # event 5: (1, 0) samples 7 + 9, then (0, 0); event 7: (0, 0), (1, 0) 2 + 3, (0, 1), (1, 1) 5 + 6
    assert ev.features[:, 0].tolist() == [16.0, 8.0, 1.0, 5.0, 4.0, 11.0]
    assert ev.features[:, 1].tolist() == [70.0, 80.0, 0.0, 10.0, 30.0, 40.0]   # x of the first sample
    np.testing.assert_allclose(ev.features[:, 4], [0.7, 0.8, 0.1, 0.2, 0.4, 0.5], rtol=1e-6)  # first sample's unc


def test_labels_row_aligned_and_checked():
    ev = load(samples="sum", label_fields=("Deposit_energy", "con_pid0", "n_contributions", "con_Energy0"),
              with_event_label=True)
    assert ev.hit_labels["con_pid0"].tolist() == [11, 22, 22, 211, 0, 13]
    assert ev.label_offsets.tolist() == [0, 2, 5]
    truth, per_event = hit_truth(ev), event_truth(ev)
    assert truth["hit_class"].tolist() == [0, 0, 0, 1, 3, 2]
    assert np.isnan(truth["log10_kinetic"][4])
    np.testing.assert_allclose(truth["log10_kinetic"][3], np.log10(3.0 - 0.13957039))
    assert per_event["n_particles"].tolist() == [2, 3]
    np.testing.assert_allclose(per_event["sum_kinetic"], [2.0 - 0.00051099895 + 4.0, 5.0 + 3.0 - 0.13957039
                                                          + 1.0 - 0.1056583755])
    try:
        load(misalign=True, label_fields=("con_pid0",))
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned labels were not detected")


def test_paper_names_read_as_aliases():
    # a file in the paper's naming loads exactly like the BIC-named one, reported under the canonical names
    fields = TRUTH_LABEL_FIELDS + ("Deposit_energy", "con_index0")
    kw = {"samples": "sum", "extra_fields": ("amplitude_unc", "x_unc"), "label_fields": fields,
          "with_event_label": True}
    bic, paper = load(**kw), load(naming="paper", **kw)
    assert paper.columns == bic.columns == ("amplitude", "x", "y", "z", "amplitude_unc", "x_unc")
    assert np.array_equal(paper.features, bic.features)
    assert sorted(paper.hit_labels) == sorted(bic.hit_labels) == sorted(fields)
    assert all(np.array_equal(paper.hit_labels[k], bic.hit_labels[k]) for k in fields)
    assert paper.event_label.dtype.names == bic.event_label.dtype.names == ("event", "particle", "Energy", "pid")
    assert all(np.array_equal(paper.event_label[k], bic.event_label[k]) for k in bic.event_label.dtype.names)
    for a, b in ((hit_truth(paper), hit_truth(bic)), (event_truth(paper), event_truth(bic))):
        assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
    # paper names read a BIC-named file, per array: labels deposit_energy is Deposit_energy (0 for the noise hit),
    # not the measurements amplitude
    alias = load(feature_fields=("deposit_energy", "x", "y", "z"),
                 label_fields=("deposit_energy", "con0_particle", "con0_pid"))
    canon = load(label_fields=("Deposit_energy", "con_index0", "con_pid0"))
    assert alias.columns == canon.columns and np.array_equal(alias.features, canon.features)
    assert sorted(alias.hit_labels) == sorted(canon.hit_labels)
    assert all(np.array_equal(alias.hit_labels[k], canon.hit_labels[k]) for k in canon.hit_labels)
    assert alias.hit_labels["Deposit_energy"].tolist() == [7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 0.0, 5.0, 6.0]
    try:
        load(label_fields=("con_pid0", "con0_pid"))
    except ValueError:
        pass
    else:
        raise AssertionError("a field requested under its canonical name and an alias was not refused")


def test_max_events_and_subsystem_filter():
    ev = load(max_events=1)
    assert ev.event_ids.tolist() == [5] and ev.offsets.tolist() == [0, 3]
    ev = load(subsystems=(1,))
    assert ev.features[:, 0].tolist() == [7.0, 9.0, 2.0, 3.0, 5.0, 6.0]


def test_augmentation_leaves_extra_columns():
    ev = load(extra_fields=("x_unc", "y_unc"))
    ds = NpzMeasurementsDataset(ev, transform=default_augmentation(drop_rate=0.3), normalize=True,
                                norm={"log_e_mean": 0.0, "log_e_std": 1.0, "pos_scale": 1000.0},
                                extra_scale=[2.0, 4.0])
    torch.manual_seed(0)
    original = ev.hits(1)
    for _ in range(20):
        v1, _ = ds[1]
        x_unc = v1.x[:, 4].numpy() * 2.0
        assert set(x_unc.tolist()) <= set(original[:, 4].tolist())     # dropped rows only, values untouched
        np.testing.assert_allclose(v1.x[:, 5].numpy() * 4.0, 2.0 * x_unc)


def test_gaussian_noise_and_normalize_unchanged_for_four_columns():
    def old_noise(pc, sigma_pos=1.0, sigma_e=0.01):
        noise_e = pc[:, 0] * torch.randn(pc.size(0)) * sigma_e
        pc[:, 0] = pc[:, 0] + noise_e
        pc[:, 1:] = pc[:, 1:] + torch.randn_like(pc[:, 1:]) * sigma_pos
        return pc

    pc = torch.rand(50, 4) * 100
    torch.manual_seed(1)
    a = old_noise(pc.clone())
    torch.manual_seed(1)
    b = GaussianNoise()(pc.clone())
    assert torch.equal(a, b)
    old_norm = torch.empty_like(pc)
    old_norm[:, 0] = (torch.log10(pc[:, 0].clamp(min=1e-6)) - (-3.3915)) / 0.4410
    old_norm[:, 1:] = pc[:, 1:] / 1000.0
    assert torch.equal(normalize_hits(pc), old_norm)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED")
