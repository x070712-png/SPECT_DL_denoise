# scripts/quantify_noisy_baseline.py
"""
"Before training" activity-recovery baseline: how well does the NOISY
INPUT (before any denoising) recover the true activity in each VOI,
compared to what the model output will later be measured against?

Needs no GPU, no checkpoint — data/dataset already has the noisy inputs
on disk. Run on the login node:

    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/quantify_noisy_baseline.py \
        --data_dir data/dataset --split val --out_csv logs/quant_noisy_baseline.csv

Reads phantom_idx/alpha pairs the same way SPECTDataset does (via
build_split), so the CSV lines up with whatever split you point it at.
"""

import argparse
import csv
import os
 
import numpy as np
 
from spect.baseline.dataset import build_split
from spect.baseline.quantification import build_voi_masks
 
 
def alpha_to_float(alpha_str):
    """Convert a folder-name-safe alpha string like '0p125' back to the
    float count-level (0.125). Reverses the 'p'-for-'.' encoding used
    throughout data/dataset's alpha_* folder names.
 
    Needed because raw RC is confounded with the known count-level scaling:
    at low alpha, the noisy input's absolute values are proportionally
    lower (fewer counts collected), so RC naturally tracks alpha even
    with a perfect reconstruction. Dividing RC by alpha removes this
    expected/known scaling, leaving only the "extra" bias caused by
    noise/reconstruction (Kris, 7/13 meeting)."""
    return float(alpha_str.replace("p", "."))
 
 
def compute_isolation_flags(per_voi):
    """For each ellipsoid in a phantom, return whether its mask has zero
    voxel overlap with every OTHER ellipsoid's mask in the same phantom.
 
    Needed because generate_ellipsoids.py stacks overlapping ellipsoids'
    intensities additively (region[inside] += intensity). A per-VOI RC
    that divides by only that ellipsoid's own intensity ends up inflated
    wherever its mask overlaps a neighbour, since the measured signal
    there includes the neighbour's contribution too but the denominator
    doesn't. This inflation hits small ellipsoids hardest (overlap is a
    bigger fraction of a small volume), which can mask or even reverse
    the true partial-volume-effect trend in the size-grouped RC summary.
    Restricting that analysis to isolated (non-overlapping) ellipsoids
    removes this confound -- see quantification.py's note on overlap
    regions, and the 7/19 discussion of why per-VOI RC/alpha came out
    >1 across the board."""
    n = len(per_voi)
    flags = [True] * n
    for i in range(n):
        for j in range(i + 1, n):
            if np.any(per_voi[i]["mask"] & per_voi[j]["mask"]):
                flags[i] = False
                flags[j] = False
    return flags
 
 
def process_phantom(phantom_idx, alpha_str, data_dir, seed_base, verbose=True,
                     input_prefix="input", eps=1e-8, label_dir=None):
    """Load one phantom's noisy input (or, with input_prefix="denoised",
    a model's restored count-domain output -- see run_inference_dump.py)
    PLUS its label (label_{idx}.npy -- the noise-free reconstruction the
    network was actually trained to reproduce), and compute recovery
    stats against BOTH reference points.
 
    UPDATED 7/26 (Cate, 7/23 meeting): the network only ever sees
    (noisy input, label) pairs during training -- it was never shown, and
    never asked to correct for, the gap between the label and the raw
    phantom ground truth (that gap is purely a property of the forward
    projection + OSEM reconstruction step, e.g. resolution blur /
    partial-volume effect). Scoring the network against ground truth
    conflates that reconstruction-only bias with the network's own
    denoising error, which is unfair to the network and doesn't isolate
    either effect cleanly. So this now computes THREE things per VOI,
    all using the exact same mask (see quantification.py -- the mask
    itself is unchanged, only which array it's applied to is new):
 
      true_val_gt    = background + intensity (phantom design value,
                        exactly as before -- see quantification.py note)
      true_val_label = label[mask].mean()  (NEW -- what the network was
                        actually trained to reproduce)
      recon_rc       = true_val_label / true_val_gt  (label vs ground
                        truth -- pure reconstruction bias, nothing to do
                        with noise or the CNN)
      mean_rc        = measured[mask].mean() / true_val_label  (measured
                        vs label -- "measured" is the noisy input when
                        input_prefix="input", or the CNN output when
                        input_prefix="denoised". THIS is the number that
                        actually reflects what the network is being asked
                        to do, and is comparable before/after the CNN)
 
    Run this script twice -- once with input_prefix="input" and once with
    input_prefix="denoised" (after run_inference_dump.py) -- and put
    mean_rc from both runs alongside recon_rc in one table: that's Cate's
    three-way comparison (label vs ground truth, reconstruction vs label,
    CNN output vs label).
 
    NOTE on alpha-normalisation: label_{idx}.npy lives inside the
    alpha_{alpha_str} folder, i.e. it's generated at that same reduced
    count level -- unlike true_val_gt (phantom design value, alpha-
    independent), true_val_label should already be on the same alpha-
    scaled footing as "measured". mean_rc_over_alpha is kept below for
    continuity with the old GT-based numbers, but check the printed
    per-alpha true_val_label values the first time you run this --
    if true_val_label scales roughly linearly with alpha, mean_rc should
    already be close to alpha-independent and dividing by alpha again
    may not be the right thing to do. Flag this rather than assume it.
 
    Returns (combined_row, per_voi_entries). combined_row is None if the
    input or label file is missing; per_voi_entries is always a list
    (possibly empty).
 
    label_dir: where to look for label_{idx}.npy. Defaults to data_dir
    (fine when data_dir IS data/dataset, i.e. input_prefix="input"), but
    MUST be set explicitly to the original data/dataset root when data_dir
    points at a run_inference_dump.py output folder (input_prefix=
    "denoised") -- that folder only ever contains denoised_{idx}.npy, it
    never copies the labels alongside them, so label lookups would
    otherwise all silently miss and every phantom gets skipped.
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
 
    measured = np.load(inp_path).astype(np.float32)
    label = np.load(label_path).astype(np.float32)
 
    combined_mask, per_voi, background = build_voi_masks(phantom_idx, seed_base=seed_base)
    alpha_val = alpha_to_float(alpha_str)
    isolation_flags = compute_isolation_flags(per_voi)
 
    # --- combined (all-VOI) recovery ---
    if combined_mask.sum() > 0:
        # true value for the combined mask: approximate as background +
        # mean ellipsoid intensity (overlap regions are an approximation
        # -- see quantification.py note). Fine for a first-pass baseline.
        mean_intensity = float(np.mean([v["intensity"] for v in per_voi])) if per_voi else 0.0
        true_val_gt = background + mean_intensity
        true_val_label = float(label[combined_mask].mean())
        measured_mean = float(measured[combined_mask].mean())
 
        recon_rc = true_val_label / (true_val_gt + eps)
        recon_bias_pct = (true_val_label - true_val_gt) / (true_val_gt + eps) * 100.0
 
        combined_mean_rc = measured_mean / (true_val_label + eps)
        combined_bias_pct = (measured_mean - true_val_label) / (true_val_label + eps) * 100.0
        # kept for continuity with the old GT-based numbers -- see NOTE
        # on alpha-normalisation in the docstring above before trusting this.
        combined_mean_rc_over_alpha = combined_mean_rc / alpha_val
    else:
        true_val_gt = true_val_label = float("nan")
        recon_rc, recon_bias_pct = float("nan"), float("nan")
        combined_mean_rc, combined_bias_pct = float("nan"), float("nan")
        combined_mean_rc_over_alpha = float("nan")
 
    combined_row = {
        "phantom_idx": phantom_idx,
        "alpha": alpha_str,
        "n_voi": len(per_voi),
        "true_val_gt": true_val_gt,
        "true_val_label": true_val_label,
        "recon_rc_label_over_gt": recon_rc,
        "recon_bias_pct": recon_bias_pct,
        "combined_mean_rc": combined_mean_rc,
        "combined_bias_pct": combined_bias_pct,
        "combined_mean_rc_over_alpha": combined_mean_rc_over_alpha,
    }
 
    # --- per-VOI recovery, so you can group by size later ---
    per_voi_entries = []
    for i, v in enumerate(per_voi):
        true_val_gt_v = background + v["intensity"]
        true_val_label_v = float(label[v["mask"]].mean())
        measured_mean_v = float(measured[v["mask"]].mean())
 
        recon_rc_v = true_val_label_v / (true_val_gt_v + eps)
        mean_rc = measured_mean_v / (true_val_label_v + eps)
        bias_pct = (measured_mean_v - true_val_label_v) / (true_val_label_v + eps) * 100.0
        mean_rc_over_alpha = mean_rc / alpha_val
 
        per_voi_entries.append({
            "phantom_idx": phantom_idx,
            "alpha": alpha_str,
            "alpha_val": alpha_val,
            "voi_idx": i,
            "mean_radius_vox": v["mean_radius_vox"],
            "n_voxels": v["n_voxels"],
            "true_val_gt": true_val_gt_v,
            "true_val_label": true_val_label_v,
            "recon_rc_label_over_gt": recon_rc_v,
            "mean_rc": mean_rc,
            "bias_pct": bias_pct,
            "mean_rc_over_alpha": mean_rc_over_alpha,
            "is_isolated": isolation_flags[i],
        })
 
    if verbose:
        tag = "CNN out" if input_prefix == "denoised" else "noisy in"
        print(f"phantom {phantom_idx:04d} alpha_{alpha_str}: "
              f"recon(label/GT)={recon_rc:.3f}  {tag}/label RC={combined_mean_rc:.3f} "
              f"(RC/alpha={combined_mean_rc_over_alpha:.3f}) bias={combined_bias_pct:+.1f}% "
              f"({len(per_voi)} VOIs)")
 
    return combined_row, per_voi_entries
 
 
def print_size_binned_summary(title, entries, n_size_bins):
    """Shared helper for the two size-binned summaries (all-VOI and
    isolated-only) -- equal-COUNT bins (terciles by default) rather than
    fixed radius thresholds, so each bin has a comparable sample size
    regardless of how the random radii happened to distribute."""
    print(f"\n=== {title} ===")
    if not entries:
        print("  (no entries)")
        return
 
    radii = np.array([r["mean_radius_vox"] for r in entries])
    edges = np.quantile(radii, np.linspace(0, 1, n_size_bins + 1))
    edges[-1] += 1e-6  # make the top edge inclusive
 
    for b in range(n_size_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [r for r in entries if lo <= r["mean_radius_vox"] < hi]
        if not in_bin:
            continue
        mean_rc_bin = np.mean([r["mean_rc"] for r in in_bin])
        mean_rc_over_alpha_bin = np.mean([r["mean_rc_over_alpha"] for r in in_bin])
        print(f"  radius [{lo:.1f}, {hi:.1f}) vox: n={len(in_bin):3d}  "
              f"mean RC={mean_rc_bin:.3f}  mean RC/alpha={mean_rc_over_alpha_bin:.3f}")
 
    return edges
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset")
    p.add_argument("--label_dir", type=str, default=None,
                    help="where label_{idx}.npy files live. Defaults to --data_dir, "
                         "which is correct when --input_prefix=input (data_dir IS "
                         "data/dataset). MUST be set explicitly to data/dataset when "
                         "--input_prefix=denoised, since run_inference_dump.py's output "
                         "folder only contains denoised_{idx}.npy, not the labels -- "
                         "e.g. --data_dir logs/denoised/3d_unet --label_dir data/dataset")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                    help="split used for the combined-mask RC-by-alpha baseline "
                         "(should match whatever split the after-denoising "
                         "comparison will use)")
    p.add_argument("--out_csv", type=str, default="logs/quant_noisy_baseline.csv")
    p.add_argument("--per_voi_csv", type=str, default=None,
                    help="where to save the per-ellipsoid breakdown "
                         "(default: same dir as --out_csv, suffixed _per_voi.csv)")
    p.add_argument("--n_size_bins", type=int, default=3,
                    help="number of equal-COUNT size bins (terciles by default) "
                         "for the per-ellipsoid-size RC summary")
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--input_prefix", type=str, default="input",
                    help="filename prefix to read, e.g. 'input' (default, the raw "
                         "noisy baseline) or 'denoised' (a checkpoint's restored "
                         "count-domain output, written by run_inference_dump.py) -- "
                         "everything else about the RC computation is identical, so "
                         "the two runs' CSVs are directly comparable")
    p.add_argument("--pool_all_for_size_analysis", action="store_true", default=None,
                    help="Pass 2 (per-ellipsoid size analysis) pools train+val+test "
                         "by default when --input_prefix=input (no model involved, no "
                         "leakage concern, more statistical power). When "
                         "--input_prefix=denoised, this defaults to OFF instead -- "
                         "pooling in train would mix in samples the model was "
                         "directly optimised on, inflating RC on those phantoms "
                         "relative to true generalisation. Pass --pool_all_for_size_analysis "
                         "to force pooling anyway (e.g. if you only care about the "
                         "reconstruction-pipeline PVE trend and are OK with the caveat).")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
 
    if args.per_voi_csv is None:
        base, ext = os.path.splitext(args.out_csv)
        args.per_voi_csv = f"{base}_per_voi{ext}"
 
    # pool_all_for_size_analysis default depends on input_prefix -- see
    # parse_args() help text: pooling train is fine with no model involved,
    # but leaks train performance into the number once a checkpoint is
    # measuring the data.
    if args.pool_all_for_size_analysis is None:
        pool_all = (args.input_prefix == "input")
    else:
        pool_all = args.pool_all_for_size_analysis
 
    # ------------------------------------------------------------------
    # Pass 1: combined-mask RC by alpha, scoped to --split. This is the
    # number that will later be compared against "after denoising" on a
    # trained checkpoint, so it stays tied to val/test, not all 500.
    # ------------------------------------------------------------------
    pairs = build_split(args.split)
    rows = []
    for phantom_idx, alpha_str in pairs:
        combined_row, _ = process_phantom(phantom_idx, alpha_str, args.data_dir,
                                           args.seed_base, verbose=True,
                                           input_prefix=args.input_prefix,
                                           label_dir=args.label_dir)
        if combined_row is not None:
            rows.append(combined_row)
 
    # ---- write a flat summary CSV (combined-level numbers) ----
    fieldnames = ["phantom_idx", "alpha", "n_voi", "true_val_gt", "true_val_label",
                  "recon_rc_label_over_gt", "recon_bias_pct", "combined_mean_rc",
                  "combined_bias_pct", "combined_mean_rc_over_alpha"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
 
    # ---- also print an overall summary, grouped by alpha ----
    tag = "CNN output" if args.input_prefix == "denoised" else "noisy input"
    print(f"\n=== Summary by alpha ({tag} vs label, split={args.split}) ===")
    alphas = sorted(set(r["alpha"] for r in rows))
    for a in alphas:
        subset = [r["combined_mean_rc"] for r in rows if r["alpha"] == a and not np.isnan(r["combined_mean_rc"])]
        subset_norm = [r["combined_mean_rc_over_alpha"] for r in rows
                        if r["alpha"] == a and not np.isnan(r["combined_mean_rc_over_alpha"])]
        subset_recon = [r["recon_rc_label_over_gt"] for r in rows
                         if r["alpha"] == a and not np.isnan(r["recon_rc_label_over_gt"])]
        subset_label = [r["true_val_label"] for r in rows if r["alpha"] == a and not np.isnan(r["true_val_label"])]
        if subset:
            print(f"  alpha_{a}: mean RC (vs label) = {np.mean(subset):.3f}  "
                  f"mean RC/alpha = {np.mean(subset_norm):.3f}  "
                  f"recon RC (label/GT) = {np.mean(subset_recon):.3f}  "
                  f"mean true_val_label = {np.mean(subset_label):.3f}  (n={len(subset)})")
    print("\n  (check mean true_val_label above across alpha groups -- if it scales "
          "roughly linearly with alpha, RC vs label is already alpha-matched and "
          "RC/alpha may be double-normalising; see docstring note in process_phantom)")
 
    # RC/alpha should be roughly flat across alpha groups if the residual
    # bias is purely count-level/noise-driven and not something else --
    # print the overall spread so it's obvious at a glance whether it is.
    all_norm = [r["combined_mean_rc_over_alpha"] for r in rows if not np.isnan(r["combined_mean_rc_over_alpha"])]
    if all_norm:
        print(f"\nRC/alpha across all groups: mean={np.mean(all_norm):.3f}, "
              f"std={np.std(all_norm):.3f}, min={np.min(all_norm):.3f}, max={np.max(all_norm):.3f}")
 
    print(f"\nSaved {len(rows)} rows to {args.out_csv}")
 
    # ------------------------------------------------------------------
    # Pass 2: per-ellipsoid size analysis, pooling ALL 500 phantoms
    # (train+val+test) -- not a model-evaluation question, so no leakage
    # concern, and more data is needed since only ~9% of ellipsoids are
    # isolated (non-overlapping). See module docstring.
    # ------------------------------------------------------------------
    if pool_all:
        print(f"\n[per-ellipsoid size analysis below uses ALL 500 phantoms "
              f"(train+val+test), not just --split {args.split} -- see 7/19 "
              f"discussion in module docstring]")
        all_pairs = build_split("train") + build_split("val") + build_split("test")
    else:
        print(f"\n[input_prefix={args.input_prefix}: per-ellipsoid size analysis "
              f"below is restricted to --split {args.split} only, NOT pooling "
              f"train, to avoid mixing in phantoms the model was directly "
              f"optimised on -- pass --pool_all_for_size_analysis to override]")
        all_pairs = build_split(args.split)
    per_voi_rows = []
    for i, (phantom_idx, alpha_str) in enumerate(all_pairs):
        _, per_voi_entries = process_phantom(phantom_idx, alpha_str, args.data_dir,
                                              args.seed_base, verbose=False,
                                              input_prefix=args.input_prefix,
                                              label_dir=args.label_dir)
        per_voi_rows.extend(per_voi_entries)
        if (i + 1) % 100 == 0:
            print(f"  ...processed {i + 1}/{len(all_pairs)} phantoms")
 
    # ---- write the flat per-VOI CSV ----
    per_voi_fieldnames = ["phantom_idx", "alpha", "alpha_val", "voi_idx",
                           "mean_radius_vox", "n_voxels", "true_val_gt", "true_val_label",
                           "recon_rc_label_over_gt", "mean_rc", "bias_pct",
                           "mean_rc_over_alpha", "is_isolated"]
    with open(args.per_voi_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_voi_fieldnames)
        writer.writeheader()
        for r in per_voi_rows:
            writer.writerow({k: r[k] for k in per_voi_fieldnames})
    print(f"Saved {len(per_voi_rows)} per-VOI rows "
          f"({'all 500 phantoms' if pool_all else f'--split {args.split} only'}) "
          f"to {args.per_voi_csv}")
 
    # ---- per-ellipsoid RC grouped by size, all VOIs (Kris, 7/13 meeting:
    # "don't just look at the combined mask number -- check whether small
    # ellipsoids recover worse than large ones") ----
    print_size_binned_summary(
        f"Per-ellipsoid RC grouped by size, ALL VOIs ({args.n_size_bins} "
        f"equal-count bins, n={len(per_voi_rows)})",
        per_voi_rows, args.n_size_bins)
 
    # ---- same bins, broken down by alpha (checks whether the size effect
    # is stable across noise levels -- constant would point to a pure
    # partial-volume/resolution effect; getting worse at low alpha would
    # suggest an interaction with noise/reconstruction instead) ----
    if per_voi_rows:
        radii = np.array([r["mean_radius_vox"] for r in per_voi_rows])
        edges = np.quantile(radii, np.linspace(0, 1, args.n_size_bins + 1))
        edges[-1] += 1e-6
        print("\n  -- same bins, broken down by alpha (checks whether the "
              "size effect is stable across noise levels) --")
        alphas = sorted(set(r["alpha"] for r in per_voi_rows))
        for a in alphas:
            line = f"  alpha_{a}: "
            parts = []
            for b in range(args.n_size_bins):
                lo, hi = edges[b], edges[b + 1]
                in_bin = [r for r in per_voi_rows
                          if r["alpha"] == a and lo <= r["mean_radius_vox"] < hi]
                if in_bin:
                    parts.append(f"bin{b}(n={len(in_bin)}) RC/alpha="
                                 f"{np.mean([r['mean_rc_over_alpha'] for r in in_bin]):.3f}")
            print(line + "  ".join(parts))
 
    # ---- same size-binned summary, but restricted to ISOLATED ellipsoids
    # only (no mask overlap with any neighbour in the same phantom) --
    # removes the additive-overlap inflation described in
    # compute_isolation_flags() above, so this should give a cleaner read
    # on the true partial-volume-effect trend. ----
    isolated_rows = [r for r in per_voi_rows if r["is_isolated"]]
    print_size_binned_summary(
        f"Same analysis, ISOLATED ellipsoids only (no overlap with any "
        f"other ellipsoid in the same phantom) -- {len(isolated_rows)}/"
        f"{len(per_voi_rows)} VOIs qualify",
        isolated_rows, args.n_size_bins)
 
 
if __name__ == "__main__":
    main()
 