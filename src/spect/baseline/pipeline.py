# src/spect/baseline/pipeline.py
"""
End-to-end baseline pipeline (WeiMiao-style) for:
template_sino -> phantom+uMap -> acquisition model -> clean sino
-> noisy sinos (alphas) -> OSEM recon (clean + noisy)

This file is intentionally "scriptable":
- can be imported as a library
- can be run as: python -m spect.baseline.pipeline

On Myriad (temporary, before packaging):
  export PYTHONPATH=$PWD/src:$PYTHONPATH
  python -m spect.baseline.pipeline --help
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import sirf.STIR as stir

from .phantom_umap import load_template_sinogram, make_phantom_and_umap
from .acquisition_model import build_ubmatrix_acq_model, forward_project, AcquisitionBundle
from .noise import make_noisy_sinos, sinogram_stats
from .recon_osem import ReconConfig, osem_reconstruct, image_stats


@dataclass
class BaselineConfig:
    # geometry / phantom
    zooms: Optional[Tuple[float, float, float]] = (0.5, 1.0, 1.0)
    mu: float = 0.12
    use_cyl_fov: bool = True

    # acquisition model
    resol_slope: float = 0.1
    resol_sigma0: float = 0.1
    full_3D: bool = False  # keep False for 2D/2.5D baseline

    # noise
    alphas: Tuple[float, ...] = (5.0, 1.0, 0.5, 0.05)
    seed: Optional[int] = 0

    # recon
    num_subsets: int = 21
    num_subiters: int = 42
    init_value: float = 1.0


@dataclass
class BaselineOutputs:
    # Core objects (SIRF)
    templ_sino: stir.AcquisitionData
    acq_bundle: AcquisitionBundle

    clean_sino: stir.AcquisitionData
    noisy_sinos: Dict[float, stir.AcquisitionData]

    recon_clean: stir.ImageData
    recon_noisy: Dict[float, stir.ImageData]

    # Quick stats for logging
    clean_sino_stats: Dict[str, float]
    noisy_sino_stats: Dict[float, Dict[str, float]]
    recon_clean_stats: Dict[str, float]
    recon_noisy_stats: Dict[float, Dict[str, float]]


def run_baseline(cfg: BaselineConfig) -> BaselineOutputs:
    """
    Run the whole baseline pipeline and return SIRF objects + stats.

    This function is the one you will call from notebooks / batch jobs.
    """
    # 1) template
    templ = load_template_sinogram()

    # 2) phantom + umap on consistent grid
    bundle = make_phantom_and_umap(
        templ,
        zooms=cfg.zooms,
        mu=cfg.mu,
        use_cyl_fov=cfg.use_cyl_fov,
    )

    # 3) acquisition model
    acq = build_ubmatrix_acq_model(
        templ_sino=templ,
        umap=bundle.umap,
        resol_slope=cfg.resol_slope,
        resol_sigma0=cfg.resol_sigma0,
        full_3D=cfg.full_3D,
    )

    # 4) forward projection -> clean sino
    clean_sino = forward_project(acq, bundle.activity, templ)
    clean_stats = sinogram_stats(clean_sino)

    # 5) noisy sinos
    noisy_dict = make_noisy_sinos(clean_sino, alphas=list(cfg.alphas), seed=cfg.seed)
    noisy_stats = {a: sinogram_stats(s) for a, s in noisy_dict.items()}

    # 6) recon (clean + noisy)
    recon_cfg = ReconConfig(
        num_subsets=cfg.num_subsets,
        num_subiters=cfg.num_subiters,
        init_value=cfg.init_value,
    )

    recon_clean = osem_reconstruct(
        clean_sino,
        acq_model=acq.acq_model,        # IMPORTANT: pass the STIR AcquisitionModel
        img_template=bundle.activity,    # geometry template
        config=recon_cfg,
        use_cyl_fov=cfg.use_cyl_fov,
    )
    recon_clean_stats = image_stats(recon_clean)

    recon_noisy: Dict[float, stir.ImageData] = {}
    recon_noisy_stats: Dict[float, Dict[str, float]] = {}

    for a in sorted(noisy_dict.keys()):
        r = osem_reconstruct(
            noisy_dict[a],
            acq_model=acq.acq_model,
            img_template=bundle.activity,
            config=recon_cfg,
            use_cyl_fov=cfg.use_cyl_fov,
        )
        recon_noisy[a] = r
        recon_noisy_stats[a] = image_stats(r)

    return BaselineOutputs(
        templ_sino=templ,
        acq_bundle=acq,
        clean_sino=clean_sino,
        noisy_sinos=noisy_dict,
        recon_clean=recon_clean,
        recon_noisy=recon_noisy,
        clean_sino_stats=clean_stats,
        noisy_sino_stats=noisy_stats,
        recon_clean_stats=recon_clean_stats,
        recon_noisy_stats=recon_noisy_stats,
    )


def _print_summary(cfg: BaselineConfig, out: BaselineOutputs) -> None:
    print("=== BaselineConfig ===")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")

    print("\n=== Clean sinogram ===")
    print("shape:", out.clean_sino.as_array().shape)
    print("stats:", out.clean_sino_stats)

    print("\n=== Noisy sinograms ===")
    for a in sorted(out.noisy_sinos.keys()):
        arr = out.noisy_sinos[a].as_array()
        print(f"alpha={a} shape={arr.shape} stats={out.noisy_sino_stats[a]}")

    print("\n=== Recon (clean) ===")
    print("shape:", out.recon_clean.as_array().shape)
    print("stats:", out.recon_clean_stats)

    print("\n=== Recon (noisy) ===")
    for a in sorted(out.recon_noisy.keys()):
        print(f"alpha={a} stats={out.recon_noisy_stats[a]}")

    print("\nSMOKE CHECK: recon_clean max/mean > 0 ?",
          out.recon_clean_stats["max"] > 0 and out.recon_clean_stats["mean"] > 0)


def _smoke_test() -> None:
    """
    Minimal end-to-end check.
    If this passes on Myriad, you basically have the full baseline chain working.
    """
    cfg = BaselineConfig()
    out = run_baseline(cfg)
    _print_summary(cfg, out)

    # Basic sanity asserts
    assert out.clean_sino_stats["max"] > 0
    assert out.recon_clean_stats["max"] > 0
    assert out.recon_clean_stats["mean"] > 0
    print("\nSMOKE TEST PASSED")


def _main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.12)
    p.add_argument("--alphas", type=str, default="5,1,0.5,0.05",
                   help="comma-separated alphas, e.g. '5,1,0.5,0.05'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subsets", type=int, default=21)
    p.add_argument("--subiters", type=int, default=42)
    p.add_argument("--init", type=float, default=1.0)
    p.add_argument("--no-zoom", action="store_true", help="disable zooms=(0.5,1,1)")
    p.add_argument("--no-cyl", action="store_true", help="disable cylindrical FOV truncation")
    args = p.parse_args()

    alphas = tuple(float(x.strip()) for x in args.alphas.split(",") if x.strip())

    cfg = BaselineConfig(
        zooms=None if args.no_zoom else (0.5, 1.0, 1.0),
        mu=args.mu,
        use_cyl_fov=not args.no_cyl,
        alphas=alphas,
        seed=args.seed,
        num_subsets=args.subsets,
        num_subiters=args.subiters,
        init_value=args.init,
    )

    out = run_baseline(cfg)
    _print_summary(cfg, out)


if __name__ == "__main__":
    # If you run as a script: python -m spect.baseline.pipeline
    _main()