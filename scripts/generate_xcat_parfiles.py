# scripts/generate_xcat_parfiles.py
"""
Generate N randomised XCAT .par files from a template general.samp.par,
following Wei Miao's own published randomisation protocol:

  "we uniformly randomised body habitus (torso long-/short-axis scaling
  0.9-1.1), cardiac size (three-axis scaling 0.9-1.1), respiratory motion
  (diaphragm excursion 1.0-3.0; anterior-posterior expansion 0.3-0.8),
  starting phases for cardiac and respiratory motion (0-1), sex
  (male/female), and organ activities (each scaled by 0.8-1.2), while
  keeping the acquisition geometry and reconstruction settings fixed."
"""

import argparse
import csv
import os
import random
import re
 
RANDOMIZED_RANGES = {
    "torso_long_axis_scale": (0.9, 1.1),
    "torso_short_axis_scale": (0.9, 1.1),
    "hrt_scale_x": (0.9, 1.1),
    "hrt_scale_y": (0.9, 1.1),
    "hrt_scale_z": (0.9, 1.1),
    "max_diaphragm_motion": (1.0, 3.0),
    "max_AP_exp": (0.3, 0.8),
    "hrt_start_ph_index": (0.0, 1.0),
    "resp_start_ph_index": (0.0, 1.0),
}
 
ACTIVITY_SCALE_RANGE = (0.8, 1.2)
 
MALE_ORGAN_FILE, MALE_HEART_BASE = "vmale50.nrb", "vmale50_heart.nrb"
FEMALE_ORGAN_FILE, FEMALE_HEART_BASE = "vfemale50.nrb", "vfemale50_heart.nrb"
 
# same 5 count levels + folder-name-safe encoding as dataset.py's GROUP_TO_ALPHA
ALPHA_LEVELS = ["1p0", "0p5", "0p25", "0p125", "0p05"]
 
# matches lines like "key = value   # comment...", tolerant of tabs/spaces.
# value is anything up to the next whitespace or '#' -- covers numbers
# (1.0, 0.442) and bare filenames (vmale50.nrb) alike.
LINE_RE = re.compile(r'^(?P<key>[A-Za-z_][A-Za-z0-9_]*)(?P<eq>\s*=\s*)(?P<value>[^\s#]+)(?P<rest>.*)$')
 
 
def is_activity_key(key):
    return key.endswith("_activity") or key.endswith("_act")
 
 
def randomize_par(template_lines, rng):
    """Return a new list of lines with the randomised keys substituted.
    Non-matching lines (including the huge NOTES comment block at the end
    of the file) are passed through byte-for-byte."""
    gender = rng.choice([0, 1])
    organ_file = FEMALE_ORGAN_FILE if gender == 1 else MALE_ORGAN_FILE
    heart_base = FEMALE_HEART_BASE if gender == 1 else MALE_HEART_BASE
 
    out = []
    seen_activity_keys = 0
    for line in template_lines:
        m = LINE_RE.match(line)
        if not m:
            out.append(line)
            continue
        key, eq, value, rest = m.group("key"), m.group("eq"), m.group("value"), m.group("rest")
 
        if key == "gender":
            out.append(f"{key}{eq}{gender}{rest}\n" if not rest.endswith("\n") else f"{key}{eq}{gender}{rest}")
        elif key == "organ_file":
            out.append(f"{key}{eq}{organ_file}{rest}\n" if not rest.endswith("\n") else f"{key}{eq}{organ_file}{rest}")
        elif key == "heart_base":
            out.append(f"{key}{eq}{heart_base}{rest}\n" if not rest.endswith("\n") else f"{key}{eq}{heart_base}{rest}")
        elif key in RANDOMIZED_RANGES:
            lo, hi = RANDOMIZED_RANGES[key]
            new_val = rng.uniform(lo, hi)
            out.append(f"{key}{eq}{new_val:.4f}{rest}\n" if not rest.endswith("\n") else f"{key}{eq}{new_val:.4f}{rest}")
        elif is_activity_key(key):
            try:
                base_val = float(value)
            except ValueError:
                out.append(line)  # e.g. atten_table_filename ends in neither suffix, shouldn't hit this branch anyway
                continue
            factor = rng.uniform(*ACTIVITY_SCALE_RANGE)
            new_val = base_val * factor
            seen_activity_keys += 1
            out.append(f"{key}{eq}{new_val:.4f}{rest}\n" if not rest.endswith("\n") else f"{key}{eq}{new_val:.4f}{rest}")
        else:
            out.append(line)
 
    return out, gender, seen_activity_keys
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--template", type=str, required=True,
                    help="path to the base general.samp.par to randomise from")
    p.add_argument("--out_dir", type=str, required=True,
                    help="where to write the N randomised .par files")
    p.add_argument("--n_phantoms", type=int, required=True,
                    help="how many phantoms to generate parfiles for -- gets split "
                         "into 5 contiguous groups (one per alpha), same "
                         "as generate_xcat_dataset.py's default grouping")
    p.add_argument("--basename_prefix", type=str, default="xcat",
                    help="dxcat2 output basename prefix -- phantom i gets basename "
                         "'{prefix}_{i:04d}', producing {prefix}_{i:04d}_act_1.bin etc.")
    p.add_argument("--manifest", type=str, required=True,
                    help="output CSV: par_path,basename,phantom_idx,alpha_str,gender")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()
 
 
def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
 
    with open(args.template) as f:
        template_lines = f.readlines()
 
    # same contiguous-group alpha assignment as generate_xcat_dataset.py's
    # auto mode, computed here up front instead so it's baked into the
    # manifest from the start (no ambiguity about file-sort order later)
    n = args.n_phantoms
    base, extra = divmod(n, len(ALPHA_LEVELS))
    alpha_assignment = []
    for group_idx, alpha_str in enumerate(ALPHA_LEVELS):
        group_size = base + (1 if group_idx < extra else 0)
        alpha_assignment.extend([alpha_str] * group_size)
 
    rows = []
    for i in range(n):
        phantom_idx = args.start_idx + i
        basename = f"{args.basename_prefix}_{phantom_idx:04d}"
        par_lines, gender, n_activity = randomize_par(template_lines, rng)
        par_path = os.path.join(args.out_dir, f"{basename}.par")
        with open(par_path, "w") as f:
            f.writelines(par_lines)
        rows.append({
            "par_path": par_path,
            "basename": basename,
            "phantom_idx": phantom_idx,
            "alpha_str": alpha_assignment[i],
            "gender": "female" if gender == 1 else "male",
        })
        if i == 0:
            print(f"[check] first phantom randomised {n_activity} activity-suffix "
                  f"keys -- sanity check this against a manual grep of the template "
                  f"for '_activity' + '_act' occurrences before trusting the rest.")
 
    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["par_path", "basename", "phantom_idx", "alpha_str", "gender"])
        writer.writeheader()
        writer.writerows(rows)
 
    counts = {}
    for r in rows:
        counts[r["alpha_str"]] = counts.get(r["alpha_str"], 0) + 1
    print(f"Wrote {n} .par files to {args.out_dir}, manifest at {args.manifest}")
    print("Per-alpha counts:", counts)
    print(f"\nNext: run_xcat_parfiles.sh to actually invoke dxcat2 on each .par "
          f"(needs the apptainer container, not this Python env).")
 
 
if __name__ == "__main__":
    main()