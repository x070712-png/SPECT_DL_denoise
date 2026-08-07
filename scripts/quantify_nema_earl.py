# scripts/quantify_nema_earl.py
"""
Recovery Coefficient (RC) quantification for the NEMA/EARL evaluation
phantoms -- same three-way comparison (label vs GT, measured vs label) as
quantify_noisy_baseline.py, but for a SINGLE fixed physical phantom with
REAL sphere masks (NEMA_sphere_10mm.npy etc.), not the ellipsoid pipeline's
seed-generated VOI masks.

Key differences from quantify_noisy_baseline.py:
  - No phantom_idx / build_split() -- NEMA/EARL are one fixed phantom, data
    lives at {data_dir}/alpha_{alpha_str}/{input,label}.npy (see
    generate_eval_phantom_dataset.py), not per-phantom-idx files.
  - VOI masks are real, fixed sphere mask files (one per sphere diameter),
    not generated from a seed -- no overlap/isolation concern, spheres in
    a NEMA/EARL IQ phantom are physically separated by design.
  - true_val_gt is read directly from the phantom's own activity map
    (--activity, e.g. nema/NEMA_activity.npy) using the SAME sphere mask,
    rather than reconstructed from background+intensity ellipsoid params.

Usage (noisy input, "before CNN"):
    module unload gcc-libs
    module load pytorch/2.1.0/gpu
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/quantify_nema_earl.py \
        --activity nema/NEMA_activity.npy \
        --data_dir data/nema_dataset \
        --sphere_dir nema \
        --sphere_prefix NEMA_sphere_ \
        --input_prefix input \
        --out_csv logs/quant_nema_baseline.csv

Usage (CNN output, "after CNN" -- after run_inference_dump_nema_earl.py
has written denoised.npy into each alpha_*/ folder):
    python3 scripts/quantify_nema_earl.py \
        --activity nema/NEMA_activity.npy \
        --data_dir logs/denoised/3d_unet_nema \
        --sphere_dir nema \
        --sphere_prefix NEMA_sphere_ \
        --input_prefix denoised \
        --out_csv logs/quant_nema_unet_output.csv

Same usage pattern for EARL, just swap nema -> earl paths.
"""

import argparse
import csv
import glob
import os
import re
 
import numpy as np
 
from spect.baseline.config import COUNT_LEVELS
 
 
ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}
ALPHA_STR_TO_FLOAT = {v: k for k, v in ALPHA_STR.items()}
 
 
def find_sphere_masks(sphere_dir, sphere_prefix):
    """Discover all {sphere_prefix}{size}mm.npy files in sphere_dir and
    return a list of (size_mm, path) sorted by size, smallest first."""
    pattern = os.path.join(sphere_dir, f"{sphere_prefix}*mm.npy")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No sphere mask files found matching {pattern}")
 
    spheres = []
    for p in paths:
        m = re.search(r"(\d+)mm\.npy$", os.path.basename(p))
        if not m:
            print(f"[warn] could not parse sphere size from {p}, skipping")
            continue
        spheres.append((int(m.group(1)), p))
 
    spheres.sort(key=lambda x: x[0])
    return spheres
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--activity", type=str, required=True,
                    help="path to the phantom's true activity map .npy "
                         "(e.g. nema/NEMA_activity.npy) -- used as ground "
                         "truth, same array generate_eval_phantom_dataset.py "
                         "forward-projected")
    p.add_argument("--data_dir", type=str, required=True,
                    help="dir with alpha_{alpha_str}/{input_prefix}_seed{seed}.npy -- "
                         "e.g. data/nema_dataset (noisy input) or "
                         "logs/denoised/3d_unet_nema (CNN output)")
    p.add_argument("--label_dir", type=str, default=None,
                    help="dir with alpha_{alpha_str}/label.npy -- defaults to "
                         "--data_dir (correct for the noisy-input/baseline case, "
                         "where label.npy lives alongside input_seed*.npy). MUST "
                         "be set explicitly to the original data/{nema,earl}_dataset "
                         "dir when --data_dir points at a CNN-output directory, "
                         "since run_inference_nema_earl.py never copies label.npy "
                         "into its --out_dir.")
    p.add_argument("--sphere_dir", type=str, required=True,
                    help="dir containing the sphere mask .npy files, "
                         "e.g. nema or earl")
    p.add_argument("--sphere_prefix", type=str, required=True,
                    help="filename prefix before the size, e.g. "
                         "'NEMA_sphere_' or 'EARL_sphere_'")
    p.add_argument("--input_prefix", type=str, default="input",
                    help="'input' for the raw noisy baseline (default), or "
                         "'denoised' for a checkpoint's restored output "
                         "(written by a NEMA/EARL inference-dump step). "
                         "Reads {input_prefix}_seed{seed}.npy for each --seeds entry.")
    p.add_argument("--alphas", type=float, nargs="+", default=COUNT_LEVELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="noise realization seeds to average RC over -- must "
                         "match what generate_eval_phantom_dataset.py was run "
                         "with (default: single seed 42, no averaging).")
    p.add_argument("--out_csv", type=str, required=True,
                    help="per-alpha, per-sphere, per-seed RC output CSV")
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()
 
 
def main():
    args = parse_args()
    label_dir = args.label_dir if args.label_dir is not None else args.data_dir
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
 
    activity = np.load(args.activity).astype(np.float32)
    print(f"Loaded activity map: {args.activity} "
          f"(shape={activity.shape}, min={activity.min():.4f}, max={activity.max():.4f})")
    print(f"data_dir (measured) = {args.data_dir}")
    print(f"label_dir (GT label) = {label_dir}")
 
    spheres = find_sphere_masks(args.sphere_dir, args.sphere_prefix)
    print(f"Found {len(spheres)} sphere masks: {[s[0] for s in spheres]} mm")
 
    sphere_masks = []
    for size_mm, path in spheres:
        mask = np.load(path).astype(bool)
        if mask.shape != activity.shape:
            raise ValueError(
                f"{path}: mask shape {mask.shape} != activity shape {activity.shape}"
            )
        n_voxels = int(mask.sum())
        true_val_gt = float(activity[mask].mean()) if n_voxels > 0 else float("nan")
        sphere_masks.append({
            "size_mm": size_mm,
            "mask": mask,
            "n_voxels": n_voxels,
            "true_val_gt": true_val_gt,
        })
        print(f"  {size_mm}mm sphere: n_voxels={n_voxels}, true_val_gt={true_val_gt:.4f}")
 
    rows = []  # per (alpha, sphere, seed) -- raw, unaveraged
    for alpha in args.alphas:
        alpha_str = ALPHA_STR.get(alpha, str(alpha).replace(".", "p"))
        measured_alpha_dir = os.path.join(args.data_dir, f"alpha_{alpha_str}")
        label_alpha_dir = os.path.join(label_dir, f"alpha_{alpha_str}")
 
        label_path = os.path.join(label_alpha_dir, "label.npy")
        if not os.path.exists(label_path):
            print(f"[skip] alpha_{alpha_str}: missing {label_path} "
                  f"(check --label_dir if --data_dir points at a CNN-output dir)")
            continue
        label = np.load(label_path).astype(np.float32)
 
        for seed in args.seeds:
            measured_path = os.path.join(measured_alpha_dir, f"{args.input_prefix}_seed{seed}.npy")
            if not os.path.exists(measured_path):
                print(f"[skip] alpha_{alpha_str} seed={seed}: missing {measured_path}")
                continue
            measured = np.load(measured_path).astype(np.float32)
 
            for sph in sphere_masks:
                mask = sph["mask"]
                n_voxels = sph["n_voxels"]
                if n_voxels == 0:
                    continue
 
                true_val_gt = sph["true_val_gt"]
                true_val_label = float(label[mask].mean())
                measured_mean = float(measured[mask].mean())
 
                recon_rc = true_val_label / (true_val_gt + args.eps)
                recon_bias_pct = (true_val_label - true_val_gt) / (true_val_gt + args.eps) * 100.0
 
                mean_rc = measured_mean / (true_val_label + args.eps)
                bias_pct = (measured_mean - true_val_label) / (true_val_label + args.eps) * 100.0
                mean_rc_over_alpha = mean_rc / alpha
 
                rows.append({
                    "sphere_mm": sph["size_mm"],
                    "alpha": alpha_str,
                    "alpha_val": alpha,
                    "seed": seed,
                    "n_voxels": n_voxels,
                    "true_val_gt": true_val_gt,
                    "true_val_label": true_val_label,
                    "measured_mean": measured_mean,
                    "recon_rc_label_over_gt": recon_rc,
                    "recon_bias_pct": recon_bias_pct,
                    "mean_rc": mean_rc,
                    "bias_pct": bias_pct,
                    "mean_rc_over_alpha": mean_rc_over_alpha,
                })
 
        tag = "CNN out" if args.input_prefix == "denoised" else "noisy in"
        for sph in sphere_masks:
            vals = [r["mean_rc"] for r in rows if r["alpha"] == alpha_str and r["sphere_mm"] == sph["size_mm"]]
            vals_norm = [r["mean_rc_over_alpha"] for r in rows if r["alpha"] == alpha_str and r["sphere_mm"] == sph["size_mm"]]
            if not vals:
                continue
            print(f"alpha_{alpha_str} sphere={sph['size_mm']}mm: "
                  f"{tag}/label RC = {np.mean(vals):.3f} +/- {np.std(vals):.3f}  "
                  f"(RC/alpha = {np.mean(vals_norm):.3f} +/- {np.std(vals_norm):.3f}, "
                  f"n_seeds={len(vals)})")
 
    fieldnames = ["sphere_mm", "alpha", "alpha_val", "seed", "n_voxels", "true_val_gt",
                  "true_val_label", "measured_mean", "recon_rc_label_over_gt",
                  "recon_bias_pct", "mean_rc", "bias_pct", "mean_rc_over_alpha"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved {len(rows)} rows (per alpha x sphere x seed) to {args.out_csv}")
 
    if not rows:
        print("\n[warn] 0 rows saved -- nothing to summarize. Check --data_dir/--label_dir above.")
        return
 
    # ---- summary by alpha (averaged across spheres AND seeds) ----
    tag = "CNN output" if args.input_prefix == "denoised" else "noisy input"
    print(f"\n=== Summary by alpha ({tag} vs label, averaged over "
          f"{len(args.seeds)} noise realization(s) x spheres) ===")
    alphas_seen = sorted(set(r["alpha"] for r in rows), key=lambda a: ALPHA_STR_TO_FLOAT.get(a, 0))
    for a in alphas_seen:
        subset = [r["mean_rc"] for r in rows if r["alpha"] == a]
        subset_norm = [r["mean_rc_over_alpha"] for r in rows if r["alpha"] == a]
        print(f"  alpha_{a}: mean RC = {np.mean(subset):.3f} +/- {np.std(subset):.3f}  "
              f"mean RC/alpha = {np.mean(subset_norm):.3f} +/- {np.std(subset_norm):.3f}  "
              f"(n={len(subset)} = n_spheres x n_seeds)")
 
    # ---- summary by sphere size (averaged across alpha AND seeds) ----
    print(f"\n=== Summary by sphere size (averaged across all alphas x "
          f"{len(args.seeds)} noise realization(s)) ===")
    sizes_seen = sorted(set(r["sphere_mm"] for r in rows))
    for sz in sizes_seen:
        subset = [r["mean_rc"] for r in rows if r["sphere_mm"] == sz]
        subset_norm = [r["mean_rc_over_alpha"] for r in rows if r["sphere_mm"] == sz]
        print(f"  {sz}mm: mean RC = {np.mean(subset):.3f} +/- {np.std(subset):.3f}  "
              f"mean RC/alpha = {np.mean(subset_norm):.3f} +/- {np.std(subset_norm):.3f}  "
              f"(n={len(subset)})")
 
 
if __name__ == "__main__":
    main()