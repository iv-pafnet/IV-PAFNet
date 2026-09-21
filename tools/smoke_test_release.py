"""Build IV-PAFNet and run one weight-free forward pass.

This checks imports, configuration expansion, model construction, CUDA execution,
and output shapes. It does not download data or pretrained weights and does not
claim to reproduce paper metrics.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rtdetr/paper_73e/DV_Full_S42.yml",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    if args.size < 160 or args.size % 32 != 0:
        parser.error("--size must be a multiple of 32 and at least 160")

    os.environ.setdefault("DATASET_ROOT", str(REPO_ROOT / "datasets-placeholder"))
    os.environ.setdefault("OUTPUT_ROOT", str(REPO_ROOT / "outputs"))

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["PResNet"]["pretrained"] = False
    cfg.yaml_cfg["HybridEncoder"]["eval_spatial_size"] = [args.size, args.size]
    cfg.yaml_cfg["RTDETRTransformer"]["eval_spatial_size"] = [args.size, args.size]
    model = cfg.model.to(args.device).eval()
    sample = torch.randn(1, 6, args.size, args.size, device=args.device)

    with torch.no_grad():
        output = model(sample)

    shapes = {
        key: tuple(value.shape)
        for key, value in output.items()
        if hasattr(value, "shape")
    }
    print(f"config: {config_path}")
    print(f"device: {args.device}")
    print(f"input: {tuple(sample.shape)}")
    print(f"outputs: {shapes}")
    print(f"total_parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        "trainable_parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )


if __name__ == "__main__":
    main()
