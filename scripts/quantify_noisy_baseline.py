# scripts/quantify_noisy_baseline.py
"""
"Before training" activity-recovery baseline: how well does the NOISY
INPUT (before any denoising) recover the true activity in each VOI,
compared to what the model output will later be measured against?

Recovery is measured against the LABEL (the noise-free reconstruction the
network was actually trained to reproduce), not the raw phantom ground
truth. The network never sees ground truth, so scoring it against
ground truth would conflate reconstruction-only bias (resolution blur /
partial-volume effect, baked into the label already) with the network's
own denoising error. See process_phantom()'s docstring for the full
three-way breakdown (ground truth vs label vs measured). Run this once
with --input_prefix input (before CNN) and once with --input_prefix
denoised (after CNN, needs run_inference_dump.py first) to get all three
legs of the comparison.

Needs no GPU, no checkpoint. Run on the login node:

    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/quantify_noisy_baseline.py \
        --data_dir data/dataset --split val --out_csv logs/quant_noisy_baseline.csv

Reads phantom_idx/alpha pairs the same way SPECTDataset does (via
build_split), so the CSV lines up with whatever split you point it at.

This script runs two passes over the dataset, because they answer two
different questions that need two different amounts of data:

  1. Combined-mask RC by alpha -- scoped to --split (val by default). This
     is the "before denoising" baseline that will later be compared
     directly against "after denoising" numbers computed on a trained
     checkpoint, so it MUST stay on the same split (val/test) that the
     model evaluation will use -- can't compare against train (the model
     saw it during training) and shouldn't silently change the split the
     baseline numbers were already reported on.

  2. Per-ellipsoid RC grouped by size (both all-VOI and isolated-only
     variants) -- pools ALL 500 phantoms (train+val+test), not just
     --split. This isn't a model-evaluation question at all -- it's
     asking whether the reconstruction pipeline itself has a structural
     partial-volume-effect trend, which is a property of the phantom
     generation + reconstruction, not of any train/val/test split. There's
     no leakage concern (no model involved), so using all 500 just gives
     more statistical power -- important since only ~9% of ellipsoids
     turn out to be non-overlapping (see compute_isolation_flags), so the
     isolated-only breakdown needs all the samples it can get.
"""

import argparse
import csv
import os

import numpy as np

from spect.baseline.dataset import build_split
from spect.baseline.quantification import build_voi_masks

ALPHAS_ORDERED = ["1p0", "0p5", "0p25", "0p125", "0p05"]


def alpha_to_float(alpha_str):
    """Convert a folder-name-safe alpha string like '0p125' back to the
    float count-level (0.125). Reverses the 'p'-for-'.' encoding used
    throughout data/dataset's alpha_* folder names.

    Needed because raw RC is confounded with the known count-level scaling:
    at low alpha, the noisy input's absolute values are proportionally
    lower (fewer counts collected), so RC naturally tracks alpha even
    with a perfect reconstruction. Dividing RC by alpha removes this
    expected/known scaling, leaving only the "extra" bias caused by
    noise/reconstruction."""
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
    regions."""
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

    The network only ever sees (noisy input, label) pairs during
    training. It was never shown, and never asked to correct for, the
    gap between the label and the raw phantom ground truth (that gap is
    purely a property of the forward projection + OSEM reconstruction
    step, e.g. resolution blur / partial-volume effect). Scoring the
    network against ground truth conflates that reconstruction-only bias
    with the network's own denoising error, which is unfair to the
    network and doesn't isolate either effect cleanly. So this computes
    THREE things per VOI, all using the exact same mask (see
    quantification.py -- the mask itself is unchanged, only which array
    it's applied to is new):

      true_val_gt    = background + intensity (phantom design value,
                        exactly as before -- see quantification.py note)
      true_val_label = label[mask].mean()  (what the network was
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
    mean_rc from both runs alongside recon_rc in one table for the full
    three-way comparison (label vs ground truth, reconstruction vs label,
    CNN output vs label).
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
        # which column to trust depends on the run -- see process_phantom
        # docstring; both are printed so the caller never has to remember.
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
                         "comparison will use). Ignored if --phantom_indices is given.")
    p.add_argument("--phantom_indices", type=str, default=None,
                    help="comma-separated phantom indices, e.g. '90,91,...,99' -- if "
                         "given, Pass 1 (combined-mask RC by alpha) uses these SAME "
                         "indices at all 5 alphas instead of --split's block-based "
                         "pairing, so the same fixed set of phantoms can be compared "
                         "like-for-like across noise realisations. Use indices 90-99 "
                         "-- alpha_1p0's existing test-split holdout, which is unseen "
                         "training data under every alpha (see run_inference_dump.py's "
                         "'FIXED-PHANTOM MODE' docstring for why). Pass 2 (per-ellipsoid "
                         "size analysis) is unaffected by this flag -- it answers a "
                         "different question (structural PVE-vs-size trend) and "
                         "still pools/restricts per --pool_all_for_size_analysis.")
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
    p.add_argument("--pool_all_for_size_analysis", type=str, default="auto",
                    choices=["auto", "yes", "no"],
                    help="Pass 2 (per-ellipsoid size analysis) pools train+val+test "
                         "by default ('auto') when --input_prefix=input (no model "
                         "involved, no leakage concern, more statistical power). When "
                         "--input_prefix=denoised, 'auto' instead defaults to OFF -- "
                         "pooling in train would mix in samples the model was "
                         "directly optimised on, inflating RC on those phantoms "
                         "relative to true generalisation. Pass 'yes' to force pooling "
                         "anyway (e.g. if you only care about the reconstruction-"
                         "pipeline PVE trend and are OK with the caveat), or 'no' to "
                         "force restricting to --split only even when --input_prefix="
                         "input -- needed to run a val-only 'before CNN' per-VOI/"
                         "isolated-ellipsoid comparison that's sample-matched against "
                         "a val-only 'after CNN' run (auto would otherwise pool all "
                         "500 phantoms here, giving the two legs different sample sets).")
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
    # measuring the data. 'yes'/'no' override the auto-default explicitly
    # in either direction (e.g. 'no' with --input_prefix=input, to force a
    # val-only 'before CNN' run that's sample-matched against a val-only
    # 'after CNN' run).
    if args.pool_all_for_size_analysis == "auto":
        pool_all = (args.input_prefix == "input")
    else:
        pool_all = (args.pool_all_for_size_analysis == "yes")

    # ------------------------------------------------------------------
    # Pass 1: combined-mask RC by alpha, scoped to --split. This is the
    # number that will later be compared against "after denoising" on a
    # trained checkpoint, so it stays tied to val/test, not all 500.
    # ------------------------------------------------------------------
    if args.phantom_indices:
        indices = [int(x) for x in args.phantom_indices.split(",")]
        pairs = [(idx, a) for idx in indices for a in ALPHAS_ORDERED]
        print(f"[FIXED-PHANTOM MODE] Pass 1 uses {len(indices)} phantom(s) {indices} "
              f"x {len(ALPHAS_ORDERED)} alphas = {len(pairs)} pairs (--split={args.split} ignored)")
    else:
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
              f"(train+val+test), not just --split {args.split} -- see module "
              f"docstring]")
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

    # ---- per-ellipsoid RC grouped by size, all VOIs -- checks whether
    # small ellipsoids recover worse than large ones, not just the
    # combined-mask number ----
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
                    # print both raw RC and RC/alpha -- which one applies
                    # depends on the run type, see process_phantom docstring
                    parts.append(f"bin{b}(n={len(in_bin)}) RC="
                                 f"{np.mean([r['mean_rc'] for r in in_bin]):.3f}"
                                 f"/RC/alpha="
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