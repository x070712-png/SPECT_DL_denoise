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
                    help="dir with alpha_{alpha_str}/{input_prefix,label}.npy "
                         "-- e.g. data/nema_dataset (noisy input / label) or "
                         "logs/denoised/3d_unet_nema (CNN output)")
    p.add_argument("--sphere_dir", type=str, required=True,
                    help="dir containing the sphere mask .npy files, "
                         "e.g. nema or earl")
    p.add_argument("--sphere_prefix", type=str, required=True,
                    help="filename prefix before the size, e.g. "
                         "'NEMA_sphere_' or 'EARL_sphere_'")
    p.add_argument("--input_prefix", type=str, default="input",
                    help="'input' for the raw noisy baseline (default), or "
                         "'denoised' for a checkpoint's restored output "
                         "(written by a NEMA/EARL inference-dump step)")
    p.add_argument("--alphas", type=float, nargs="+", default=COUNT_LEVELS)
    p.add_argument("--out_csv", type=str, required=True,
                    help="per-alpha, per-sphere RC output CSV")
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    activity = np.load(args.activity).astype(np.float32)
    print(f"Loaded activity map: {args.activity} "
          f"(shape={activity.shape}, min={activity.min():.4f}, max={activity.max():.4f})")

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

    rows = []
    for alpha in args.alphas:
        alpha_str = ALPHA_STR.get(alpha, str(alpha).replace(".", "p"))
        alpha_dir = os.path.join(args.data_dir, f"alpha_{alpha_str}")

        measured_path = os.path.join(alpha_dir, f"{args.input_prefix}.npy")
        label_path = os.path.join(alpha_dir, "label.npy")

        if not os.path.exists(measured_path):
            print(f"[skip] alpha_{alpha_str}: missing {measured_path}")
            continue
        if not os.path.exists(label_path):
            print(f"[skip] alpha_{alpha_str}: missing {label_path}")
            continue

        measured = np.load(measured_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)

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

            row = {
                "sphere_mm": sph["size_mm"],
                "alpha": alpha_str,
                "alpha_val": alpha,
                "n_voxels": n_voxels,
                "true_val_gt": true_val_gt,
                "true_val_label": true_val_label,
                "measured_mean": measured_mean,
                "recon_rc_label_over_gt": recon_rc,
                "recon_bias_pct": recon_bias_pct,
                "mean_rc": mean_rc,
                "bias_pct": bias_pct,
                "mean_rc_over_alpha": mean_rc_over_alpha,
            }
            rows.append(row)

            tag = "CNN out" if args.input_prefix == "denoised" else "noisy in"
            print(f"alpha_{alpha_str} sphere={sph['size_mm']}mm: "
                  f"recon(label/GT)={recon_rc:.3f}  {tag}/label RC={mean_rc:.3f} "
                  f"(RC/alpha={mean_rc_over_alpha:.3f}) bias={bias_pct:+.1f}%")

    fieldnames = ["sphere_mm", "alpha", "alpha_val", "n_voxels", "true_val_gt",
                  "true_val_label", "measured_mean", "recon_rc_label_over_gt",
                  "recon_bias_pct", "mean_rc", "bias_pct", "mean_rc_over_alpha"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved {len(rows)} rows to {args.out_csv}")

    # ---- summary by alpha (averaged across spheres) ----
    tag = "CNN output" if args.input_prefix == "denoised" else "noisy input"
    print(f"\n=== Summary by alpha ({tag} vs label) ===")
    alphas_seen = sorted(set(r["alpha"] for r in rows), key=lambda a: ALPHA_STR_TO_FLOAT.get(a, 0))
    for a in alphas_seen:
        subset = [r["mean_rc"] for r in rows if r["alpha"] == a]
        subset_norm = [r["mean_rc_over_alpha"] for r in rows if r["alpha"] == a]
        print(f"  alpha_{a}: mean RC = {np.mean(subset):.3f}  "
              f"mean RC/alpha = {np.mean(subset_norm):.3f}  (n_spheres={len(subset)})")

    # ---- summary by sphere size (averaged across alpha) ----
    print(f"\n=== Summary by sphere size (averaged across all alphas) ===")
    sizes_seen = sorted(set(r["sphere_mm"] for r in rows))
    for sz in sizes_seen:
        subset = [r["mean_rc"] for r in rows if r["sphere_mm"] == sz]
        subset_norm = [r["mean_rc_over_alpha"] for r in rows if r["sphere_mm"] == sz]
        print(f"  {sz}mm: mean RC = {np.mean(subset):.3f}  "
              f"mean RC/alpha = {np.mean(subset_norm):.3f}  (n_alphas={len(subset)})")


if __name__ == "__main__":
    main()