# src/spect/baseline/generate_xcat_dataset.py
#
# Lives alongside generate_dataset.py / generate_ellipsoids.py / sirf_bridge.py

"""
Run XCAT activity phantoms through the SAME forward-projection + OSEM
reconstruction pipeline used for the 500 synthetic ellipsoid phantoms
(sirf_bridge.py -- acquire_data() + reconstruct_data()), producing
input_NNNN.npy / label_NNNN.npy pairs in the same layout as data/dataset
(alpha_{alpha_str}/{input,label}_{idx:04d}.npy) -- so
quantify_noisy_baseline.py, run_inference_dump.py etc. can point at this
output dir exactly the same way they point at data/dataset.
"""
 
import argparse
import csv
import glob
import os
 
import numpy as np
 
from src.spect.baseline.sirf_bridge import (
    load_template_sinogram,
    acquire_data,
    reconstruct_data,
)
 
# same 5 count levels + folder-name-safe encoding as the ellipsoid dataset
# (config.py COUNT_LEVELS, dataset.py GROUP_TO_ALPHA) -- SAME ORDER, so
# contiguous-groups mode lines up with the existing convention.
ALPHA_LEVELS = [
    (1.0, "1p0"),
    (0.5, "0p5"),
    (0.25, "0p25"),
    (0.125, "0p125"),
    (0.05, "0p05"),
]
 
VOLUME_SHAPE = (128, 128, 128)  # matches PHANTOM_CONFIG["volume_shape"]
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("task_idx", type=int, nargs="?", default=None,
                    help="0-based SGE array task index (SGE_TASK_ID - 1). If given, "
                         "only processes the slice [task_idx*phantoms_per_task : "
                         "+phantoms_per_task] of the FULL phantom list -- same "
                         "convention as generate_dataset.py. Omit for a one-off "
                         "serial run over everything.")
    p.add_argument("--phantoms_per_task", type=int, default=5,
                    help="rows processed per SGE task -- default 5 matches "
                         "generate_dataset.py / submit_dataset.sh's -t 1-100 for "
                         "500 phantoms. If you generate a different --n_phantoms "
                         "in generate_xcat_parfiles.py, recompute "
                         "submit_generate_xcat_dataset.sh's -t range as "
                         "ceil(n_phantoms/phantoms_per_task) to match.")
    p.add_argument("--xcat_dir", type=str, default=None,
                    help="directory containing XCAT *_act_*.bin activity phantoms "
                         "-- ignored if --manifest is given")
    p.add_argument("--pattern", type=str, default="*_act_*.bin",
                    help="glob pattern (relative to --xcat_dir) for activity phantom "
                         "files -- matches dxcat2's own naming, NOT *_atn_*.bin "
                         "(attenuation, intentionally unused -- see module docstring)")
    p.add_argument("--manifest", type=str, default=None,
                    help="CSV with 'phantom_path,alpha_str' per row (alpha_str one "
                         "of 1p0/0p5/0p25/0p125/0p05) -- this is generate_xcat_"
                         "parfiles.py's manifest after the awk conversion printed "
                         "by run_xcat_parfiles.sh")
    p.add_argument("--out_dir", type=str, default="data/xcat_dataset",
                    help="root output dir -- gets alpha_{alpha_str}/ subfolders, "
                         "same layout as data/dataset")
    p.add_argument("--start_idx", type=int, default=0,
                    help="phantom_idx to start numbering from -- keep above the "
                         "ellipsoid dataset's 0-499 range if you ever want to point "
                         "the SAME quantify_noisy_baseline.py --data_dir at a merged "
                         "folder; default 0 is fine for a standalone xcat_dataset dir")
    p.add_argument("--seed", type=int, default=42,
                    help="seed for the Poisson noise draw -- only affects "
                         "reproducibility of the noisy sinogram, not the phantom "
                         "itself (XCAT phantoms are already fixed files on disk)")
    return p.parse_args()
 
 
def load_xcat_activity(path):
    """Load one XCAT *_act_*.bin file as a (128,128,128) float32 array.
    See module docstring for why no reshape/transpose is needed -- dxcat2's
    own log confirms slice-sequential C-order raw write, no header."""
    arr = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(VOLUME_SHAPE))
    if arr.size != expected:
        raise ValueError(
            f"{path}: got {arr.size} float32 elements, expected {expected} "
            f"(={'x'.join(map(str, VOLUME_SHAPE))}) -- this file is NOT a "
            f"128^3 XCAT activity phantom as-is; check dxcat2's own log for "
            f"this run to see what shape it actually wrote, don't guess-reshape."
        )
    return arr.reshape(VOLUME_SHAPE)
 
 
def build_all_phantom_alpha_pairs(args):
    """Return [(phantom_path, alpha_str, phantom_idx), ...] for EVERY
    phantom (not just this task's slice) -- phantom_idx is each row's
    position in the full list (+ start_idx). Computing the FULL list first
    and slicing afterward (in main()) is what keeps phantom_idx stable
    across SGE tasks -- deriving phantom_idx from enumerate() over a
    per-task slice would give every task phantom_idx 0..phantoms_per_task-1
    and silently overwrite each other's output files."""
    if args.manifest:
        rows = []
        valid = {a for _, a in ALPHA_LEVELS}
        with open(args.manifest, newline="") as f:
            for row in csv.DictReader(f):
                alpha_str = row["alpha_str"].strip()
                if alpha_str not in valid:
                    raise ValueError(f"manifest row has alpha_str={alpha_str!r}, "
                                      f"must be one of {sorted(valid)}")
                rows.append((row["phantom_path"].strip(), alpha_str))
    else:
        if not args.xcat_dir:
            raise SystemExit("Need either --manifest or --xcat_dir")
        act_paths = sorted(glob.glob(os.path.join(args.xcat_dir, args.pattern)))
        if not act_paths:
            raise SystemExit(f"No files matched {os.path.join(args.xcat_dir, args.pattern)}")
 
        # contiguous-groups: split act_paths into 5 groups (as equal as possible,
        # extras go to the earliest groups), group order = ALPHA_LEVELS order --
        # mirrors GROUP_TO_ALPHA's "first 100 -> alpha 1.0, next 100 -> alpha 0.5, ..."
        n = len(act_paths)
        base, extra = divmod(n, len(ALPHA_LEVELS))
        rows = []
        i = 0
        for group_idx, (_, alpha_str) in enumerate(ALPHA_LEVELS):
            group_size = base + (1 if group_idx < extra else 0)
            for path in act_paths[i:i + group_size]:
                rows.append((path, alpha_str))
            i += group_size
 
    return [(path, alpha_str, args.start_idx + i) for i, (path, alpha_str) in enumerate(rows)]
 
 
def process_one(phantom_idx, act_path, alpha_str, templ_sino, out_dir):
    alpha_val = dict((a, v) for v, a in ALPHA_LEVELS)[alpha_str]
    out_subdir = os.path.join(out_dir, f"alpha_{alpha_str}")
    os.makedirs(out_subdir, exist_ok=True)
    label_path = os.path.join(out_subdir, f"label_{phantom_idx:04d}.npy")
    input_path = os.path.join(out_subdir, f"input_{phantom_idx:04d}.npy")
 
    if os.path.exists(label_path) and os.path.exists(input_path):
        print(f"[{phantom_idx:04d}] alpha_{alpha_str} already exists, skipping.")
        return
 
    phantom = load_xcat_activity(act_path)
    print(f"[{phantom_idx:04d}] loaded {act_path}, alpha_{alpha_str}, "
          f"shape={phantom.shape}, mean={phantom.mean():.4f}, max={phantom.max():.4f}")
 
    clean_sino, noisy_sino = acquire_data(phantom, templ_sino, alpha=alpha_val)
    label_img = reconstruct_data(clean_sino, templ_sino)
    input_img = reconstruct_data(noisy_sino, templ_sino)
 
    np.save(label_path, label_img.as_array().astype(np.float32))
    np.save(input_path, input_img.as_array().astype(np.float32))
    print(f"[{phantom_idx:04d}] done -> {input_path}, {label_path}")
 
 
def main():
    args = parse_args()
    np.random.seed(args.seed)
 
    all_pairs = build_all_phantom_alpha_pairs(args)
    if not all_pairs:
        raise SystemExit("No (phantom, alpha) pairs to process.")
 
    if args.task_idx is None:
        my_pairs = all_pairs
        log_suffix = "serial"
    else:
        start = args.task_idx * args.phantoms_per_task
        end = start + args.phantoms_per_task
        my_pairs = all_pairs[start:end]
        log_suffix = str(args.task_idx)
        if not my_pairs:
            print(f"task_idx={args.task_idx}: slice [{start}:{end}] is empty "
                  f"(only {len(all_pairs)} phantoms total) -- nothing to do. "
                  f"Expected if your -t range in submit_generate_xcat_dataset.sh "
                  f"overshoots ceil(n_phantoms/phantoms_per_task).")
            return
 
    os.makedirs("logs", exist_ok=True)
    import sirf.STIR as spect
    spect.MessageRedirector(
        f"logs/xcat_info_{log_suffix}.txt",
        f"logs/xcat_warnings_{log_suffix}.txt",
        f"logs/xcat_errors_{log_suffix}.txt",
    )
 
    templ_sino = load_template_sinogram()
    print(f"Task {log_suffix}: {len(my_pairs)} phantom(s) of {len(all_pairs)} total")
    for path, alpha_str, phantom_idx in my_pairs:
        print(f"  phantom_idx {phantom_idx:04d}  alpha_{alpha_str}  <-  {path}")
 
    for path, alpha_str, phantom_idx in my_pairs:
        process_one(phantom_idx, path, alpha_str, templ_sino, args.out_dir)
 
    print(f"Task {log_suffix} complete. Output layout matches data/dataset -- "
          f"point quantify_noisy_baseline.py / run_inference_dump.py at "
          f"--data_dir {args.out_dir} the same way. NOTE: with fewer than 500 "
          f"phantoms you'll need your own build_split()-style train/val/test "
          f"partition for fine-tuning -- dataset.py's assumes 80/10/10 out of "
          f"100 per alpha group, won't fit a smaller XCAT set as-is.")
 
 
if __name__ == "__main__":
    main()