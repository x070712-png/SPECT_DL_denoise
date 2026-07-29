#!/bin/bash
# scripts/run_xcat_parfiles.sh
#
# Loops dxcat2 over every .par file generate_xcat_parfiles.py wrote, inside
# the apptainer container. Bind pattern CONFIRMED from your own manual run:
#   apptainer shell --bind /home/ucapiuw/SPECT_DL_denoise/xcat/XCAT_latest:/xcat ubuntu2204.sif
#   cd /xcat
#   ./dxcat2_linux_64bit
# i.e. XCAT_latest (host) is bind-mounted to /xcat (container), and
# dxcat2_linux_64bit lives directly inside XCAT_latest. Swapped "shell" for
# "exec" below (non-interactive, loop-friendly) and pass parfile + output
# basename as the two args dxcat2 normally takes -- CHECK this matches how
# test1 actually ran if it printed an interactive prompt instead.
#
# Assumes generate_xcat_parfiles.py's --out_dir was xcat/XCAT_latest/parfiles
# (i.e. somewhere UNDER XCAT_latest, so the container's bind mount can see
# the .par files at /xcat/parfiles/*.par) -- if you used a different
# --out_dir outside XCAT_latest, move the files first or the container
# won't see them.
#
# Usage:
#   bash scripts/run_xcat_parfiles.sh xcat_phantom_manifest.csv

set -euo pipefail

MANIFEST="${1:?usage: run_xcat_parfiles.sh <manifest_csv>}"

# --- CONFIRM these two ---
XCAT_HOST_DIR="/home/ucapiuw/SPECT_DL_denoise/xcat/XCAT_latest"  # host path bound to /xcat
CONTAINER_SIF="ubuntu2204.sif"   # .sif path -- give the full path here if this script
                                   # isn't run from the same directory you tested test1 from
# --------------------------

OUT_DIR_HOST="$XCAT_HOST_DIR/generated"
mkdir -p "$OUT_DIR_HOST"

n=0
tail -n +2 "$MANIFEST" | while IFS=, read -r par_path basename phantom_idx alpha_str gender; do
    n=$((n + 1))
    # par_path in the manifest is a host path (e.g. xcat/XCAT_latest/parfiles/xcat_0000.par)
    # -- convert to the path as seen INSIDE the container, relative to /xcat
    rel_par_path="${par_path#"$XCAT_HOST_DIR"/}"
    echo "[$n] $basename (alpha_$alpha_str, $gender) <- $rel_par_path"

    apptainer exec --bind "$XCAT_HOST_DIR":/xcat "$CONTAINER_SIF" \
        bash -c "cd /xcat && ./dxcat2_linux_64bit '$rel_par_path' 'generated/$basename'"

    if [ ! -f "$OUT_DIR_HOST/${basename}_act_1.bin" ]; then
        echo "  [WARN] expected $OUT_DIR_HOST/${basename}_act_1.bin not found -- check dxcat2 output above"
    fi
done

echo "Done. Activity phantoms in $OUT_DIR_HOST/*_act_1.bin"
echo ""
echo "If this takes too long on the login node for 500 phantoms, it may need"
echo "its own qsub wrapper -- time a handful first and decide."
echo ""
echo "Next: build the manifest generate_xcat_dataset.py needs (phantom_path,alpha_str):"
echo "  awk -F, -v d=\"$OUT_DIR_HOST\" 'NR==1{print \"phantom_path,alpha_str\"; next}"
echo "  {print d\"/\"\$2\"_act_1.bin,\"\$4}' $MANIFEST > xcat_dataset_manifest.csv"