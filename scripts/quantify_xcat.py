# scripts/quantify_xcat.py
"""
Recovery Coefficient (RC) quantification for XCAT test-split phantoms --
same three-way comparison (label vs ground truth, measured vs label) as
quantify_noisy_baseline.py, but VOI masks come from the REAL per-tissue
activity values baked into each phantom's ORIGINAL XCAT activity map
(xcat/XCAT_latest/generated/xcat_{idx:04d}_act_1.bin, looked up via
xcat_dataset_manifest.csv), NOT from regenerating a random ellipsoid via
a seed.

quantify_noisy_baseline.py's VOI masks come from build_voi_masks() -> 
get_phantom_ellipsoids() ->
generate_ellipsoids.generate_phantom(seed=seed_base+phantom_idx, ...),
which REGENERATES A FRESH, UNRELATED RANDOM ELLIPSOID PHANTOM at that
phantom_idx. Pointing --data_dir at data/xcat_dataset (or an XCAT
checkpoint's denoised output) does NOT make this valid -- the VOI
positions/shapes come from an ellipsoid geometry that has nothing to do
with the real XCAT anatomy stored on disk at that same index. Any RC
numbers previously computed this way (e.g.
quant_unet_xcat_labelalpha_output_test.csv) are INVALID and should be
discarded, not just re-organised.

VERIFIED: XCAT's raw activity phantom IS piecewise-constant by
tissue/organ -- e.g. phantom_idx=90 has only 58 unique values across the
whole 128^3 volume (0.0 = background/air, ~79.7% of voxels; 57 distinct
nonzero tissue/organ activity concentrations, from
xcat/XCAT_latest/generated/xcat_0090_act_1.bin). So each unique nonzero
value directly defines one anatomically REAL VOI, and its exact
ground-truth activity concentration IS that value -- no background+
intensity approximation needed (unlike the ellipsoid script, where
true_val_gt is only an approximation for overlap regions). This is
actually a MORE exact ground truth than the ellipsoid script has.

phantom_idx -> raw activity file mapping comes from --manifest (the same
xcat_dataset_manifest.csv used by generate_xcat_dataset.py), NOT a seed:
phantom_idx is the manifest row's 0-based position (matches
generate_xcat_dataset.py's build_all_phantom_alpha_pairs(), which assigns
phantom_idx = row index + start_idx, start_idx=0 for the standalone
xcat_dataset -- confirm this still holds if --start_idx was ever
non-default when the dataset was generated).

Usage (noisy input, "before CNN"):
    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/quantify_xcat.py \
        --manifest xcat_dataset_manifest.csv \
        --data_dir data/xcat_dataset \
        --split test \
        --input_prefix input \
        --out_csv logs/quant_xcat_noisy_baseline_test.csv

Usage (CNN output, "after CNN" -- after run_inference_dump.py; NOTE
--label_dir REQUIRED since the denoised output dir only has
denoised_NNNN.npy, not labels, same convention as quantify_noisy_baseline.py):
    python3 scripts/quantify_xcat.py \
        --manifest xcat_dataset_manifest.csv \
        --data_dir logs/denoised/3d_unet_xcat_finetune_label_alpha \
        --label_dir data/xcat_dataset \
        --split test \
        --input_prefix denoised \
        --out_csv logs/quant_xcat_unet_labelalpha_output_test.csv

Run once per checkpoint (old method + label-alpha, both architectures)
plus once for the noisy-input baseline, same pattern as the ellipsoid
3-way-comparison workflow.
"""

import argparse
import csv
import os

import numpy as np

from spect.baseline.dataset import build_split

VOLUME_SHAPE = (128, 128, 128)


def load_manifest(manifest_path):
    """Return {phantom_idx: act_bin_path}. phantom_idx is the 0-based row
    position in the manifest (excluding header) -- matches
    generate_xcat_dataset.py's build_all_phantom_alpha_pairs() numbering
    for the default --start_idx=0 case."""
    paths = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            paths[i] = row["phantom_path"].strip()
    return paths


def load_xcat_activity(path):
    """Same convention as generate_xcat_dataset.py's load_xcat_activity()
    -- 128^3 float32, no header, C-order slice-sequential."""
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(VOLUME_SHAPE))
    if arr.size != expected:
        raise ValueError(
            f"{path}: got {arr.size} float32 elements, expected {expected} "
            f"(={'x'.join(map(str, VOLUME_SHAPE))})."
        )
    return arr.reshape(VOLUME_SHAPE)


def process_phantom(phantom_idx, alpha_str, data_dir, manifest_paths, verbose=True,
                     input_prefix="input", eps=1e-8, label_dir=None, min_voi_voxels=5):
    """Load one phantom's noisy input (or CNN output) + label, plus its
    REAL raw XCAT activity map (for real per-tissue VOI masks + exact
    ground truth), and compute the same three-way comparison as
    quantify_noisy_baseline.py's process_phantom() -- see that function's
    docstring for the recon_rc / mean_rc / mean_rc_over_alpha definitions,
    identical here except true_val_gt is now an EXACT value (the raw
    activity concentration itself) rather than an approximation.
    """
    if label_dir is None:
        label_dir = data_dir
    inp_path = os.path.join(data_dir, f"alpha_{alpha_str}", f"{input_prefix}_{phantom_idx:04d}.npy")
    label_path = os.path.join(label_dir, f"alpha_{alpha_str}", f"label_{phantom_idx:04d}.npy")
    if not os.path.exists(inp_path):
        print(f"[skip] phantom {phantom_idx:04d} alpha_{alpha_str}: missing {inp_path}")
        return None, []
    if not os.path.exists(label_path):
        print(f"[skip] phantom {phantom_idx:04d} alpha_{alpha_str}: missing {label_path}")
        return None, []

    act_path = manifest_paths.get(phantom_idx)
    if act_path is None:
        print(f"[skip] phantom {phantom_idx:04d}: not found in --manifest")
        return None, []
    if not os.path.exists(act_path):
        print(f"[skip] phantom {phantom_idx:04d}: manifest points at missing file {act_path}")
        return None, []

    measured = np.load(inp_path).astype(np.float32)
    label = np.load(label_path).astype(np.float32)
    raw_act = load_xcat_activity(act_path)

    combined_mask = raw_act > 0
    alpha_val = float(alpha_str.replace("p", "."))

    if combined_mask.sum() == 0:
        print(f"[skip] phantom {phantom_idx:04d}: no nonzero-activity voxels in raw XCAT phantom")
        return None, []

    # --- combined (all-tissue) recovery -- EXACT true_val_gt, no
    # background+intensity approximation needed (unlike ellipsoids) ---
    true_val_gt = float(raw_act[combined_mask].mean())
    true_val_label = float(label[combined_mask].mean())
    measured_mean = float(measured[combined_mask].mean())

    recon_rc = true_val_label / (true_val_gt + eps)
    recon_bias_pct = (true_val_label - true_val_gt) / (true_val_gt + eps) * 100.0
    combined_mean_rc = measured_mean / (true_val_label + eps)
    combined_bias_pct = (measured_mean - true_val_label) / (true_val_label + eps) * 100.0
    combined_mean_rc_over_alpha = combined_mean_rc / alpha_val

    combined_row = {
        "phantom_idx": phantom_idx,
        "alpha": alpha_str,
        "n_voxels_nonzero": int(combined_mask.sum()),
        "true_val_gt": true_val_gt,
        "true_val_label": true_val_label,
        "recon_rc_label_over_gt": recon_rc,
        "recon_bias_pct": recon_bias_pct,
        "combined_mean_rc": combined_mean_rc,
        "combined_bias_pct": combined_bias_pct,
        "combined_mean_rc_over_alpha": combined_mean_rc_over_alpha,
    }

    # --- per-tissue-VOI recovery, so you can group by VOI size (voxel
    # count) later, same spirit as quantify_noisy_baseline.py's per-
    # ellipsoid-size breakdown -- here "size" is real tissue-region size,
    # not a randomised ellipsoid radius ---
    per_voi_entries = []
    values, counts = np.unique(raw_act[combined_mask], return_counts=True)
    for v, n in zip(values, counts):
        if n < min_voi_voxels:
            continue
        voi_mask = (raw_act == v)
        true_val_label_v = float(label[voi_mask].mean())
        measured_mean_v = float(measured[voi_mask].mean())
        true_val_gt_v = float(v)

        recon_rc_v = true_val_label_v / (true_val_gt_v + eps)
        mean_rc = measured_mean_v / (true_val_label_v + eps)
        bias_pct = (measured_mean_v - true_val_label_v) / (true_val_label_v + eps) * 100.0
        mean_rc_over_alpha = mean_rc / alpha_val

        per_voi_entries.append({
            "phantom_idx": phantom_idx,
            "alpha": alpha_str,
            "alpha_val": alpha_val,
            "true_val_gt": true_val_gt_v,
            "n_voxels": int(n),
            "true_val_label": true_val_label_v,
            "recon_rc_label_over_gt": recon_rc_v,
            "mean_rc": mean_rc,
            "bias_pct": bias_pct,
            "mean_rc_over_alpha": mean_rc_over_alpha,
        })

    if verbose:
        tag = "CNN out" if input_prefix == "denoised" else "noisy in"
        print(f"phantom {phantom_idx:04d} alpha_{alpha_str}: "
              f"recon(label/GT)={recon_rc:.3f}  {tag}/label RC={combined_mean_rc:.3f} "
              f"(RC/alpha={combined_mean_rc_over_alpha:.3f}) bias={combined_bias_pct:+.1f}% "
              f"({len(per_voi_entries)} tissue VOIs >= {min_voi_voxels} vox, "
              f"{int(combined_mask.sum())} nonzero voxels total)")

    return combined_row, per_voi_entries


def print_size_binned_summary(title, entries, n_size_bins):
    """Equal-COUNT bins by n_voxels (real tissue-region size), same
    equal-count-bin approach as quantify_noisy_baseline.py's version, just
    keyed on n_voxels directly instead of a randomised ellipsoid radius."""
    
    print(f"\n=== {title} ===")
    if not entries:
        print("  (no entries)")
        return

    sizes = np.array([r["n_voxels"] for r in entries])
    edges = np.quantile(sizes, np.linspace(0, 1, n_size_bins + 1))
    edges[-1] += 1e-6

    for b in range(n_size_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [r for r in entries if lo <= r["n_voxels"] < hi]
        if not in_bin:
            continue
        mean_rc_bin = np.mean([r["mean_rc"] for r in in_bin])
        mean_rc_over_alpha_bin = np.mean([r["mean_rc_over_alpha"] for r in in_bin])
        print(f"  n_voxels [{lo:.0f}, {hi:.0f}): n={len(in_bin):4d}  "
              f"mean RC={mean_rc_bin:.3f}  mean RC/alpha={mean_rc_over_alpha_bin:.3f}")

    return edges


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True,
                    help="xcat_dataset_manifest.csv -- maps phantom_idx (0-based row "
                         "position) to the original XCAT *_act_*.bin activity file")
    p.add_argument("--data_dir", type=str, default="data/xcat_dataset")
    p.add_argument("--label_dir", type=str, default=None,
                    help="where label_{idx}.npy files live -- defaults to --data_dir "
                         "(correct for --input_prefix=input). MUST be set explicitly to "
                         "data/xcat_dataset when --input_prefix=denoised.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--per_voi_csv", type=str, default=None,
                    help="default: same dir as --out_csv, suffixed _per_voi.csv")
    p.add_argument("--n_size_bins", type=int, default=3)
    p.add_argument("--min_voi_voxels", type=int, default=5,
                    help="skip tissue regions smaller than this (noisy per-voxel-mean "
                         "estimate otherwise) -- default 5")
    p.add_argument("--input_prefix", type=str, default="input",
                    help="'input' (raw noisy baseline) or 'denoised' (CNN output, "
                         "written by run_inference_dump.py)")
    p.add_argument("--pool_all_for_size_analysis", type=str, default="auto",
                    choices=["auto", "yes", "no"],
                    help="'auto' (default): pool train+val+test for the per-VOI size "
                         "analysis when --input_prefix=input (no model, no leakage "
                         "concern); restrict to --split only when --input_prefix=denoised "
                         "(avoid leaking train-phantom performance into the summary). "
                         "Same logic as quantify_noisy_baseline.py.")
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.per_voi_csv is None:
        base, ext = os.path.splitext(args.out_csv)
        args.per_voi_csv = f"{base}_per_voi{ext}"

    manifest_paths = load_manifest(args.manifest)
    print(f"Loaded manifest: {len(manifest_paths)} phantom_idx -> activity-file entries")

    if args.pool_all_for_size_analysis == "auto":
        pool_all = (args.input_prefix == "input")
    else:
        pool_all = (args.pool_all_for_size_analysis == "yes")

    # ------------------------------------------------------------------
    # Pass 1: combined-mask RC by alpha, scoped to --split.
    # ------------------------------------------------------------------
    pairs = build_split(args.split)
    rows = []
    for phantom_idx, alpha_str in pairs:
        combined_row, _ = process_phantom(phantom_idx, alpha_str, args.data_dir,
                                           manifest_paths, verbose=True,
                                           input_prefix=args.input_prefix,
                                           label_dir=args.label_dir,
                                           min_voi_voxels=args.min_voi_voxels)
        if combined_row is not None:
            rows.append(combined_row)

    fieldnames = ["phantom_idx", "alpha", "n_voxels_nonzero", "true_val_gt", "true_val_label",
                  "recon_rc_label_over_gt", "recon_bias_pct", "combined_mean_rc",
                  "combined_bias_pct", "combined_mean_rc_over_alpha"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    tag = "CNN output" if args.input_prefix == "denoised" else "noisy input"
    print(f"\n=== Summary by alpha ({tag} vs label, split={args.split}) ===")
    alphas = sorted(set(r["alpha"] for r in rows))
    for a in alphas:
        subset = [r["combined_mean_rc"] for r in rows if r["alpha"] == a]
        subset_norm = [r["combined_mean_rc_over_alpha"] for r in rows if r["alpha"] == a]
        subset_recon = [r["recon_rc_label_over_gt"] for r in rows if r["alpha"] == a]
        if subset:
            print(f"  alpha_{a}: mean RC (vs label) = {np.mean(subset):.3f}  "
                  f"mean RC/alpha = {np.mean(subset_norm):.3f}  "
                  f"recon RC (label/GT) = {np.mean(subset_recon):.3f}  (n={len(subset)})")

    all_norm = [r["combined_mean_rc_over_alpha"] for r in rows]
    if all_norm:
        print(f"\nRC/alpha across all groups: mean={np.mean(all_norm):.3f}, "
              f"std={np.std(all_norm):.3f}, min={np.min(all_norm):.3f}, max={np.max(all_norm):.3f}")

    print(f"\nSaved {len(rows)} rows to {args.out_csv}")

    # ------------------------------------------------------------------
    # Pass 2: per-tissue-VOI size analysis.
    # ------------------------------------------------------------------
    if pool_all:
        print(f"\n[per-VOI size analysis below uses ALL 500 phantoms (train+val+test), "
              f"not just --split {args.split} -- input_prefix=input, no model involved]")
        all_pairs = build_split("train") + build_split("val") + build_split("test")
    else:
        print(f"\n[input_prefix={args.input_prefix}: per-VOI size analysis below is "
              f"restricted to --split {args.split} only, to avoid mixing in phantoms "
              f"the model was directly optimised on]")
        all_pairs = build_split(args.split)

    per_voi_rows = []
    for i, (phantom_idx, alpha_str) in enumerate(all_pairs):
        _, per_voi_entries = process_phantom(phantom_idx, alpha_str, args.data_dir,
                                              manifest_paths, verbose=False,
                                              input_prefix=args.input_prefix,
                                              label_dir=args.label_dir,
                                              min_voi_voxels=args.min_voi_voxels)
        per_voi_rows.extend(per_voi_entries)
        if (i + 1) % 100 == 0:
            print(f"  ...processed {i + 1}/{len(all_pairs)} phantoms")

    per_voi_fieldnames = ["phantom_idx", "alpha", "alpha_val", "true_val_gt", "n_voxels",
                           "true_val_label", "recon_rc_label_over_gt", "mean_rc", "bias_pct",
                           "mean_rc_over_alpha"]
    with open(args.per_voi_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_voi_fieldnames)
        writer.writeheader()
        for r in per_voi_rows:
            writer.writerow({k: r[k] for k in per_voi_fieldnames})
    print(f"Saved {len(per_voi_rows)} per-tissue-VOI rows "
          f"({'all 500 phantoms' if pool_all else f'--split {args.split} only'}) "
          f"to {args.per_voi_csv}")

    print_size_binned_summary(
        f"Per-tissue-VOI RC grouped by voxel count ({args.n_size_bins} equal-count "
        f"bins, n={len(per_voi_rows)})",
        per_voi_rows, args.n_size_bins)


if __name__ == "__main__":
    main()