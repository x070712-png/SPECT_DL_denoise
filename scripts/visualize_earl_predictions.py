# scripts/visualize_earl_predictions.py
"""
Qualitative comparison grid for the EARL physical phantom, one row per
count level (alpha), same 5-column layout as visualize_predictions.py
(Noisy input | Noisy-label diff | Model output | Output-label diff |
Ground truth label) -- but for a SINGLE FIXED phantom across all 5 alphas
(not a different representative ellipsoid per alpha), since EARL is one
physical geometry evaluated at every count level.

Slice selection: EARL's spheres sit off-centre (ring_z=-37mm per the
phantomgen dict), so the central slice of the volume would likely miss
them. Instead this script finds the z-slice that maximises total sphere
mask coverage (summed across all 6 EARL_sphere_*mm.npy masks) and uses
that as the cross-section -- this is the "cross section over the spheres"
Kris asked for at the 8/3 meeting.

One representative seed is used for display (default 42) -- averaging
across seeds would blur the noise texture that's actually relevant to
show here; the RC numbers (from quantify_nema_earl.py) already give the
proper seed-averaged quantitative result, this plot is for visual
inspection only.

Same alpha_correction handling as visualize_predictions.py: label×alpha
checkpoints' raw saved output is ~true_scale*alpha, so it's divided by
alpha before plotting/diffing. Only label×alpha checkpoints are used for
EARL (per Stathis's 8/3 guidance -- old method isn't being evaluated on
this data), so EARL_CHECKPOINTS below only has label×alpha entries.

Usage:
    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/visualize_earl_predictions.py --checkpoint_key unet_xcat_labelalpha
    python3 scripts/visualize_earl_predictions.py --checkpoint_key swin_xcat_labelalpha
    python3 scripts/visualize_earl_predictions.py --checkpoint_key all
"""

import argparse
import os
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}
ALPHAS_ORDERED = [1.0, 0.5, 0.25, 0.125, 0.05]
 
SPHERE_DIAMETERS_MM = [13, 17, 22, 28, 37, 60]
 
# EARL data (noisy input / label) lives at data/earl_dataset_v2 -- the
# joint-calibrated (sphere_conc, background_conc) phantom that fixed the
# domain-gap bug (v1's RC/alpha was ~0.166; v2 is ~1.0-1.2). Only the
# denoised_dir + alpha_correction flag change per checkpoint.
DATA_DIR = "data/earl_dataset_v2"
SPHERE_DIR = "data/earl_phantom_v2"
SPHERE_PREFIX = "EARL_sphere_"
 
EARL_CHECKPOINTS = {
    "unet_xcat_labelalpha": {
        "label": "U-Net (XCAT finetune, label x alpha)",
        "denoised_dir": "logs/denoised/3d_unet_xcat_labelalpha_earl_v2",
        "alpha_correction": True,
    },
    "swin_xcat_labelalpha": {
        "label": "Swin UNETR (XCAT finetune, label x alpha)",
        "denoised_dir": "logs/denoised/swin_xcat_labelalpha_earl_v2",
        "alpha_correction": True,
    },
}
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_key", type=str, default="all",
                    choices=list(EARL_CHECKPOINTS.keys()) + ["all"])
    p.add_argument("--seed", type=int, default=42,
                    help="which single noise realization to display (RC numbers "
                         "are seed-averaged separately via quantify_nema_earl.py -- "
                         "this is for visual inspection of one representative draw)")
    p.add_argument("--out_dir", type=str, default="logs/qualitative_earl")
    p.add_argument("--vmax_headroom", type=float, default=1.2)
    return p.parse_args()
 
 
def find_sphere_slice():
    """Pick the z-slice with the most total sphere-mask coverage (summed
    across all 6 spheres) -- this is the EARL equivalent of central_slice()
    in visualize_predictions.py, but accounts for the spheres sitting
    off-centre (ring_z=-37mm) instead of assuming the volume's midpoint."""
    total = None
    for d in SPHERE_DIAMETERS_MM:
        mask_path = os.path.join(SPHERE_DIR, f"{SPHERE_PREFIX}{d}mm.npy")
        mask = np.load(mask_path).astype(np.int32)
        total = mask if total is None else total + mask
    per_slice_counts = total.sum(axis=(1, 2))  # sum over (H, W) per z-slice
    z = int(np.argmax(per_slice_counts))
    print(f"Selected z-slice {z} (total sphere voxels in-slice = {per_slice_counts[z]})")
    return z
 
 
def load_triplet(denoised_dir, alpha_str, seed):
    inp_path = os.path.join(DATA_DIR, f"alpha_{alpha_str}", f"input_seed{seed}.npy")
    lbl_path = os.path.join(DATA_DIR, f"alpha_{alpha_str}", "label.npy")
    den_path = os.path.join(denoised_dir, f"alpha_{alpha_str}", f"denoised_seed{seed}.npy")
 
    missing = [p for p in (inp_path, lbl_path, den_path) if not os.path.exists(p)]
    if missing:
        return None, missing
 
    inp = np.load(inp_path).astype(np.float32)
    lbl = np.load(lbl_path).astype(np.float32)
    den = np.load(den_path).astype(np.float32)
    return (inp, lbl, den), []
 
 
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
 
    keys = list(EARL_CHECKPOINTS.keys()) if args.checkpoint_key == "all" else [args.checkpoint_key]
 
    z = find_sphere_slice()
 
    # ---- Pass 1: load everything, compute global scales (same design as
    # visualize_predictions.py -- separate pre-CNN / post-CNN diff scales,
    # shared intensity scale, all computed BEFORE any plotting) ----
    loaded = {}
    all_intensity_vals = []
    all_diff_pre_vals = []
    all_diff_post_vals = []
 
    for key in keys:
        denoised_dir = EARL_CHECKPOINTS[key]["denoised_dir"]
        loaded[key] = {}
        for alpha in ALPHAS_ORDERED:
            alpha_str = ALPHA_STR[alpha]
            data, missing = load_triplet(denoised_dir, alpha_str, args.seed)
            if data is None:
                print(f"[skip] {key} alpha_{alpha_str} seed={args.seed}: missing {missing}")
                loaded[key][alpha_str] = None
                continue
 
            inp, lbl, den = data
            # raw noisy reconstruction is naturally ~alpha x dimmer (fewer
            # total counts went into OSEM) -- divide by alpha for display,
            # uniformly across EVERY checkpoint (not gated by
            # alpha_correction below), so it is comparable to the always-
            # full-scale label (Kris, 8/3 meeting; see module docstring)
            inp = inp / alpha
            if EARL_CHECKPOINTS[key]["alpha_correction"]:
                den = den / alpha
            loaded[key][alpha_str] = (inp, lbl, den)
 
            inp_s, lbl_s, den_s = inp[z], lbl[z], den[z]
            all_intensity_vals.append(inp_s.max())
            all_intensity_vals.append(lbl_s.max())
            all_intensity_vals.append(den_s.max())
            all_diff_post_vals.append(np.abs(den_s - lbl_s).max())
            all_diff_pre_vals.append(np.abs(inp_s - lbl_s).max())
 
    if not all_intensity_vals:
        raise RuntimeError("Nothing loaded -- check EARL_CHECKPOINTS paths and that "
                            "run_inference_nema_earl.py has been run for the requested "
                            "checkpoint(s) and seed.")
 
    global_vmax = max(all_intensity_vals) * args.vmax_headroom
    global_diff_pre_absmax = max(all_diff_pre_vals)
    global_diff_post_absmax = max(all_diff_post_vals)
    print(f"Global intensity vmax = {global_vmax:.3f}")
    print(f"Pre-CNN diff scale  = +/-{global_diff_pre_absmax:.3f}")
    print(f"Post-CNN diff scale = +/-{global_diff_post_absmax:.3f}")
 
    # ---- Pass 2: plot ----
    col_titles = ["Noisy input\n(/ alpha)", "(Noisy/alpha) - label\n(pre-CNN baseline)",
                  "Model output", "Output - label\n(post-CNN)", "Ground truth (label)"]
 
    for key in keys:
        n_rows = sum(1 for v in loaded[key].values() if v is not None)
        if n_rows == 0:
            print(f"[skip figure] {key}: no data loaded")
            continue
 
        fig, axes = plt.subplots(n_rows, 5, figsize=(22, 4.2 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]
 
        row = 0
        for alpha in ALPHAS_ORDERED:
            alpha_str = ALPHA_STR[alpha]
            entry = loaded[key][alpha_str]
            if entry is None:
                continue
            inp, lbl, den = entry
            inp_s, lbl_s, den_s = inp[z], lbl[z], den[z]
            diff_pre = inp_s - lbl_s
            diff_post = den_s - lbl_s
 
            axes[row, 0].imshow(inp_s, cmap="gray", vmin=0, vmax=global_vmax)
            im_pre = axes[row, 1].imshow(diff_pre, cmap="coolwarm",
                                          vmin=-global_diff_pre_absmax, vmax=global_diff_pre_absmax)
            axes[row, 2].imshow(den_s, cmap="gray", vmin=0, vmax=global_vmax)
            im_post = axes[row, 3].imshow(diff_post, cmap="coolwarm",
                                           vmin=-global_diff_post_absmax, vmax=global_diff_post_absmax)
            im_lbl = axes[row, 4].imshow(lbl_s, cmap="gray", vmin=0, vmax=global_vmax)
 
            axes[row, 0].set_ylabel(f"alpha={alpha}\nEARL, z={z}, seed={args.seed}",
                                     fontsize=10, rotation=0, labelpad=70, va="center")
 
            if row == 0:
                for c, title in enumerate(col_titles):
                    axes[row, c].set_title(title, fontsize=11)
 
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
 
            row += 1
 
        fig.colorbar(im_lbl, ax=axes[:, 4].tolist(), fraction=0.02, pad=0.02,
                     label="count-domain activity")
        fig.colorbar(im_pre, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02,
                     label="pre-CNN difference")
        fig.colorbar(im_post, ax=axes[:, 3].tolist(), fraction=0.02, pad=0.02,
                     label="post-CNN difference")
 
        fig.suptitle(f"{EARL_CHECKPOINTS[key]['label']} -- EARL phantom, seed={args.seed}, "
                      f"z={z} (cross-section through spheres); shared intensity scale across "
                      f"all rows; noisy input shown divided by alpha for display, so it is on "
                      f"the same scale as the always-full-scale label; pre-CNN and post-CNN "
                      f"diffs each on their own shared scale",
                      fontsize=12)
        out_path = os.path.join(args.out_dir, f"qualitative_earl_{key}_seed{args.seed}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
 
 
if __name__ == "__main__":
    main()