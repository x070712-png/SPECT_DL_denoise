# scripts/visualize_predictions.py
"""
Qualitative comparison grid across ALL 5 count levels, for one checkpoint
at a time -- rewritten per Kris/Cate's latest feedback plus the new
label-alpha vs old-method / U-Net vs Swin comparison need:

  1. Whole-image difference maps (no VOI-mask restriction) -- diff panels
     are computed on the whole central slice, never masked, so background
     bias is visible (Cate, 7/23 meeting).
  2. A SINGLE shared colour scale across EVERY panel/row/figure (not
     recomputed per-row, and not even per-checkpoint) -- computed once in
     a first pass over ALL data before any plotting, so brightness and
     error magnitude are directly comparable across samples, count
     levels, AND checkpoints (Cate, 7/23 meeting: "put them on the same
     scale").
  3. One representative test-split phantom per alpha level (5 rows), not
     just alpha_1p0 -- ellipsoid phantoms only exist at ONE alpha each
     (500 phantoms / 5 fixed groups), so "more count levels" means picking
     a different representative phantom per alpha, not the same phantom
     repeated at different alphas (that's only possible for NEMA/EARL).
  4. A pre-CNN baseline difference panel (noisy input - label) alongside
     the post-CNN one (model output - label), so the reader sees at a
     glance how much error the network actually removed (Cate, 7/23
     meeting).

Reads ALREADY-DUMPED denoised outputs (written by run_inference_dump.py,
the same step quantify_noisy_baseline.py's RC numbers were computed from)
instead of loading the model and running inference itself -- guarantees
the qualitative figure and the quantitative RC table agree, and this
script has no torch/model dependency (safe on the login node, no GPU
needed).

VOI-mask-overlay panel from the older version is DROPPED here -- that was
a separate methodology check (Chris's question about how the mask is
built/reused), not part of this whole-image background-bias comparison;
keep it as a separate, one-off figure if still needed.

Run ONCE PER CHECKPOINT (see CHECKPOINTS config below -- edit paths if
yours differ), or all four in one go:

    export PYTHONPATH=src:$PYTHONPATH
    python3 scripts/visualize_predictions.py --checkpoint_key unet_old
    python3 scripts/visualize_predictions.py --checkpoint_key unet_label_alpha
    python3 scripts/visualize_predictions.py --checkpoint_key swin_old
    python3 scripts/visualize_predictions.py --checkpoint_key swin_label_alpha
    python3 scripts/visualize_predictions.py --checkpoint_key all   # 4 PNGs, one global scale
"""

import argparse
import os
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from spect.baseline.dataset import build_split
 
ALPHA_STR = {1.0: "1p0", 0.5: "0p5", 0.25: "0p25", 0.125: "0p125", 0.05: "0p05"}
ALPHAS_ORDERED = [1.0, 0.5, 0.25, 0.125, 0.05]
 
# ---- EDIT THESE if your denoised-dump directories are named differently ----
# each entry: denoised .npy files live at {denoised_dir}/alpha_*/denoised_NNNN.npy
# NOTE: unet_old was previously dumped with --split val, not test -- if the
# test-split denoised_NNNN.npy files for that checkpoint don't exist yet,
# rerun run_inference_dump.py with --split test first.
#
# alpha_correction: label-alpha checkpoints were trained to predict
# label*alpha (see dataset.py's scale_label_by_alpha), so their RAW saved
# output (as dumped by run_inference_dump.py, unchanged on disk -- same
# convention quantify_noisy_baseline.py relies on for its own RC/alpha
# column) is approximately true_scale*alpha, not true_scale. Comparing
# that raw output directly against the un-scaled label makes the diff
# panels look like the network gets WORSE as alpha shrinks, when that's
# just the known label-scaling artifact, not a real regression (same trap
# already handled in quantify_noisy_baseline.py's RC/alpha column). Set
# True here for label-alpha checkpoints so this script divides by alpha
# right after loading, before computing anything else -- old-method
# checkpoints are untouched (already alpha-independent by design).
CHECKPOINTS = {
    "unet_old":          {"label": "U-Net (old method)",        "denoised_dir": "logs/denoised/3d_unet",                     "alpha_correction": False},
    "unet_label_alpha":  {"label": "U-Net (label x alpha)",      "denoised_dir": "logs/denoised/3d_unet_label_alpha",         "alpha_correction": True},
    "swin_old":          {"label": "Swin UNETR (old method)",    "denoised_dir": "logs/denoised/swin_unetr",                  "alpha_correction": False},
    "swin_label_alpha":  {"label": "Swin UNETR (label x alpha)", "denoised_dir": "logs/denoised/swin_unetr_label_alpha",      "alpha_correction": True},
}
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/dataset",
                    help="root dir with alpha_*/{input,label}_NNNN.npy (noisy input + GT label)")
    p.add_argument("--checkpoint_key", type=str, default="all",
                    choices=list(CHECKPOINTS.keys()) + ["all"])
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", type=str, default="logs/qualitative")
    p.add_argument("--slice_axis", type=int, default=0, help="0=axial, 1=coronal, 2=sagittal")
    p.add_argument("--vmax_headroom", type=float, default=1.2,
                    help="multiply the global label max by this factor for the shared "
                         "intensity colour scale (Kris, 7/15 meeting)")
    return p.parse_args()
 
 
def central_slice(vol, axis):
    idx = vol.shape[axis] // 2
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]
 
 
def pick_representative_phantoms(split):
    """One (phantom_idx, alpha_str) per alpha level, from the given split
    -- first phantom encountered in each alpha group, consistent choice
    across all checkpoints since it only depends on data_dir's split, not
    on which checkpoint is being visualised."""
    pairs = build_split(split)
    picked = {}
    for phantom_idx, alpha_str in pairs:
        if alpha_str not in picked:
            picked[alpha_str] = phantom_idx
    return picked  # {alpha_str: phantom_idx}
 
 
def load_triplet(data_dir, denoised_dir, phantom_idx, alpha_str):
    """Load (noisy input, GT label, model output) for one (phantom, alpha),
    all still in raw count-domain units (as saved on disk -- no extra
    normalisation needed here, run_inference_dump.py already restored
    count-domain units before saving denoised_NNNN.npy)."""
    inp_path = os.path.join(data_dir, f"alpha_{alpha_str}", f"input_{phantom_idx:04d}.npy")
    lbl_path = os.path.join(data_dir, f"alpha_{alpha_str}", f"label_{phantom_idx:04d}.npy")
    den_path = os.path.join(denoised_dir, f"alpha_{alpha_str}", f"denoised_{phantom_idx:04d}.npy")
 
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
 
    keys = list(CHECKPOINTS.keys()) if args.checkpoint_key == "all" else [args.checkpoint_key]
 
    representative = pick_representative_phantoms(args.split)
    print(f"Representative phantoms ({args.split} split): {representative}")
 
    # ------------------------------------------------------------------
    # Pass 1: load everything for every checkpoint x alpha up front, so a
    # SINGLE global colour scale (intensity + diff) can be computed across
    # the whole deliverable before any plotting happens.
    # ------------------------------------------------------------------
    loaded = {}   # {checkpoint_key: {alpha_str: (phantom_idx, inp, lbl, den) or None}}
    all_intensity_vals = []
    all_diff_pre_vals = []   # noisy input - label (pre-CNN), own shared scale
    all_diff_post_vals = []  # model output - label (post-CNN), own shared scale
 
    for key in keys:
        denoised_dir = CHECKPOINTS[key]["denoised_dir"]
        loaded[key] = {}
        for alpha in ALPHAS_ORDERED:
            alpha_str = ALPHA_STR[alpha]
            phantom_idx = representative.get(alpha_str)
            if phantom_idx is None:
                print(f"[warn] no {args.split}-split phantom found for alpha_{alpha_str}, skipping")
                loaded[key][alpha_str] = None
                continue
 
            data, missing = load_triplet(args.data_dir, denoised_dir, phantom_idx, alpha_str)
            if data is None:
                print(f"[skip] {key} alpha_{alpha_str} phantom {phantom_idx:04d}: missing {missing}")
                loaded[key][alpha_str] = None
                continue
 
            inp, lbl, den = data
            if CHECKPOINTS[key]["alpha_correction"]:
                # undo the label*alpha training-target scaling baked into this
                # checkpoint's raw output -- see CHECKPOINTS comment above
                den = den / alpha
            loaded[key][alpha_str] = (phantom_idx, inp, lbl, den)
 
            inp_s = central_slice(inp, args.slice_axis)
            lbl_s = central_slice(lbl, args.slice_axis)
            den_s = central_slice(den, args.slice_axis)
            all_intensity_vals.append(inp_s.max())
            all_intensity_vals.append(lbl_s.max())
            all_intensity_vals.append(den_s.max())
            all_diff_post_vals.append(np.abs(den_s - lbl_s).max())
            all_diff_pre_vals.append(np.abs(inp_s - lbl_s).max())
 
    if not all_intensity_vals:
        raise RuntimeError("Nothing loaded -- check --data_dir / CHECKPOINTS paths and that "
                            "run_inference_dump.py has been run (with --split matching --split "
                            "here) for the requested checkpoint(s).")
 
    # ---- global, shared colour scales (Cate, 7/23 meeting: single scale
    # across ALL samples, not recomputed per row/sample/checkpoint) ----
    #
    # NOTE: pre-CNN (noisy - label) and post-CNN (output - label) diffs use
    # TWO SEPARATE shared scales, not one. Pre-CNN errors are an order of
    # magnitude larger than post-CNN residuals (that's the whole point of
    # denoising) -- forcing both onto one scale washes the post-CNN panel
    # out to near-white regardless of real differences between checkpoints,
    # which would hide the label-alpha vs old-method comparison rather than
    # show it. Comparability is preserved where it matters: ALL checkpoints
    # x ALL alphas still share the SAME post-CNN scale as each other (and
    # separately, the same pre-CNN scale as each other), so old-method vs
    # label-alpha and U-Net vs Swin remain apples-to-apples. Just don't
    # shrink this further than "make the post-CNN panel legible" -- that
    # would start exaggerating residual error rather than revealing it.
    global_vmax = max(all_intensity_vals) * args.vmax_headroom
    global_diff_pre_absmax = max(all_diff_pre_vals)
    global_diff_post_absmax = max(all_diff_post_vals)
    print(f"Global intensity vmax = {global_vmax:.3f}")
    print(f"Pre-CNN diff scale  = +/-{global_diff_pre_absmax:.3f}")
    print(f"Post-CNN diff scale = +/-{global_diff_post_absmax:.3f}")
 
    # ------------------------------------------------------------------
    # Pass 2: plot, one figure per checkpoint, one row per alpha, using
    # the global scales computed above throughout.
    # ------------------------------------------------------------------
    col_titles = ["Noisy input", "Noisy - label\n(pre-CNN baseline)",
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
            phantom_idx, inp, lbl, den = entry
 
            inp_s = central_slice(inp, args.slice_axis)
            lbl_s = central_slice(lbl, args.slice_axis)
            den_s = central_slice(den, args.slice_axis)
            diff_pre = inp_s - lbl_s
            diff_post = den_s - lbl_s
 
            axes[row, 0].imshow(inp_s, cmap="gray", vmin=0, vmax=global_vmax)
            im_pre = axes[row, 1].imshow(diff_pre, cmap="coolwarm",
                                          vmin=-global_diff_pre_absmax, vmax=global_diff_pre_absmax)
            axes[row, 2].imshow(den_s, cmap="gray", vmin=0, vmax=global_vmax)
            im_post = axes[row, 3].imshow(diff_post, cmap="coolwarm",
                                           vmin=-global_diff_post_absmax, vmax=global_diff_post_absmax)
            im_lbl = axes[row, 4].imshow(lbl_s, cmap="gray", vmin=0, vmax=global_vmax)
 
            axes[row, 0].set_ylabel(f"alpha={alpha}\nphantom {phantom_idx:04d}",
                                     fontsize=10, rotation=0, labelpad=60, va="center")
 
            if row == 0:
                for c, title in enumerate(col_titles):
                    axes[row, c].set_title(title, fontsize=11)
 
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
 
            row += 1
 
        # one shared colorbar per column type (intensity, pre-CNN diff,
        # post-CNN diff), figure-wide -- pre/post diff colorbars use
        # DIFFERENT scales (see note above global_diff_pre_absmax), so both
        # need their own colorbar rather than sharing one.
        fig.colorbar(im_lbl, ax=axes[:, 4].tolist(), fraction=0.02, pad=0.02,
                     label="count-domain activity")
        fig.colorbar(im_pre, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.02,
                     label="pre-CNN difference")
        fig.colorbar(im_post, ax=axes[:, 3].tolist(), fraction=0.02, pad=0.02,
                     label="post-CNN difference")
 
        fig.suptitle(f"{CHECKPOINTS[key]['label']} -- {args.split} split, "
                      f"shared intensity scale across all rows/checkpoints; "
                      f"pre-CNN and post-CNN diffs each on their own shared "
                      f"scale (see colorbars)", fontsize=12)
        out_path = os.path.join(args.out_dir, f"qualitative_{key}_{args.split}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
 
 
if __name__ == "__main__":
    main()