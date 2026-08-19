# scripts/hpc/

SGE/`qsub` job submission scripts for UCL's Myriad HPC cluster. Each file
is a thin wrapper. It sets resource requests (`#$ -l ...`), loads the
right environment (PyTorch or SIRF/STIR modules), then calls one of the
Python entry points in `scripts/`, or the `dxcat2` binary for XCAT
generation, with the right flags. The Python files hold the actual logic
(see `scripts/README.md`). These `.sh` files just say how to run that on
the cluster; no computation happens in them directly.

Submit any of them with `qsub <path>`, and monitor with
`qstat -u $(whoami)`. Each script's own header comment states its
expected runtime and what it produces.

The subfolders are organised by pipeline stage, in the order you would
actually run them:

- `generate_data/`: produces the ellipsoidal, XCAT, and EARL datasets by
  calling `generate_dataset.py`, `generate_xcat_dataset.py`,
  `generate_eval_phantom_dataset.py`, or `dxcat2` via
  `run_xcat_parfiles.sh`. Several EARL variants exist
  (`v2`, `v3_bg0`, `v3_bg_ratio10`) because the background and
  sphere-ratio definition changed during the project. Check each
  script's header to see which variant it produces.
- `train/`: trains a checkpoint from scratch, either 3D U-Net or Swin
  UNETR, with either the old method or the label x alpha target.
- `finetune/`: fine-tunes an ellipsoid-pretrained checkpoint on the XCAT
  dataset. There is one script per architecture and training-method
  combination.
- `inference/`: runs a trained checkpoint over a dataset and dumps its
  denoised output to `.npy`, which feeds into `scripts/quantify_*.py`
  and `scripts/visualize_*.py`. This is the largest subfolder, because
  every architecture, dataset, and training-method combination that was
  actually evaluated needs its own submission script. A qsub script
  cannot take that as a runtime flag the way the underlying Python
  script can, since resource requests and environment modules have to
  be fixed at submission time.
- `visualize/`: runs the qualitative comparison scripts on a GPU node.
  This is needed because the checkpoint has to be loaded, even though
  the plotting and quantification scripts in `scripts/` run fine on the
  login node once the `.npy` files have already been dumped.

Files within each folder follow the pattern
`submit_<action>_<arch>_<dataset>_<variant>.sh`. For example,
`submit_inference_dump_swin_xcat_labelalpha.sh` is an inference-dump job
for Swin UNETR, on the XCAT dataset, using the label x alpha checkpoint.
Not every combination needs every field. `submit_train.sh` has no
dataset or variant suffix, since it is the single from-scratch U-Net run
on the ellipsoidal dataset.